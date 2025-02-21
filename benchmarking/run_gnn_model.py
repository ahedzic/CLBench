import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
from scipy.sparse import csr_matrix

class RUN_GNN(torch.nn.Module):
    def __init__(self, n_layer, hidden_dim, attn_dim, n_rel, dropout):
        super(RUN_GNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.n_rel = n_rel
        #acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = nn.ReLU()#acts[params.act]
        self.uniform_parm=False
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(
                G_GAT_Layer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.n_extra_layer = 4
        self.extra_gnn_layers = []
        for i in range(self.n_extra_layer):
            self.extra_gnn_layers.append(
                G_GAT_Layer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.extra_gnn_layers = nn.ModuleList(self.extra_gnn_layers)

        self.dropout = nn.Dropout(dropout)
        self.W_final = nn.Linear(
            self.hidden_dim, 1, bias=False)  
        self.gate = QRFGU(self.hidden_dim, self.hidden_dim)

    def get_neighbors(self, nodes, KG, M_sub, n_ent):
        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        return tail_nodes, sampled_edges, old_nodes_new_idx

    def forward(self, subs, rels, n_ent, KG, given_subs, device):
        n = len(subs)

        #n_ent = self.loader.n_ent if mode == 'transductive' else self.loader.n_ent_ind

        q_sub = torch.LongTensor(subs).to(device)
        q_rel = torch.LongTensor(rels).to(device)

        h0 = torch.zeros((n, self.hidden_dim)).to(device)
        nodes = torch.cat([torch.arange(n).unsqueeze(1).to(device), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).to(device)

        scores_all = []
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.get_neighbors(nodes.data.cpu().numpy(), KG, given_subs, n_ent)
            if i<(self.n_layer-1):
                hidden, h_n_qr = self.gnn_layers[0](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            else:
                hidden, h_n_qr = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            if torch.cuda.is_available():
                h0 = torch.zeros(nodes.size(0), hidden.size(1)).cuda().index_copy_(0, old_nodes_new_idx, h0)
            else:
                h0 = torch.zeros(nodes.size(0), hidden.size(1)).index_copy_(0, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            
            hidden = self.gate(hidden, h_n_qr, h0)
            h0 = hidden
        for i in range(self.n_extra_layer):
            
            hidden = hidden[old_nodes_new_idx]
            hidden, h_n_qr = self.extra_gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            hidden = self.dropout(hidden)
            
            hidden = self.gate(hidden, h_n_qr, h0)
            h0 = hidden

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).to(device)
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all

class G_GAT_Layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x: x):
        super(G_GAT_Layer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.relu = nn.ReLU()
        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.gate = QRFGU(in_dim, in_dim)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        unique_edges, inverse_indices = torch.unique(edges[:, [0, 2, 4]],dim=0, sorted=True, return_inverse=True)
        sub = unique_edges[:, 2]
        rel = unique_edges[:, 1]
        obj = edges[:, 5]
        h_s = hidden[sub]
        h_r = self.rela_embed(rel)
        r_idx = unique_edges[:, 0]

        h_qr = self.rela_embed(q_rel)[r_idx]

        node_group = torch.zeros(n_node, dtype=torch.int64).to(h_qr.device)
        node_group[edges[:, 5]] = edges[:, 0]  
        h_n_qr = self.rela_embed(q_rel)[node_group]  

        unique_relation_identity = self.gate(h_r, h_qr, h_s)

        unique_message = unique_relation_identity
        unique_attend_weight = self.w_alpha(self.relu(self.Ws_attn(unique_message) + self.Wqr_attn(h_qr)))

        unique_exp_attend = torch.exp(unique_attend_weight)
        exp_attend = unique_exp_attend[inverse_indices]
        unique_message = unique_exp_attend * unique_message
        message = unique_message[inverse_indices]
        sum_exp_attend = scatter(exp_attend, dim=0, index=obj, dim_size=n_node, reduce="sum")
        no_attend_message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        message_agg = no_attend_message_agg / sum_exp_attend

        hidden_new = self.act(self.W_h(message_agg))
        return hidden_new, h_n_qr

class QRFGU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, query_r: torch.Tensor, hidden_state: torch.Tensor):
        """

        :param message: message[batch_size,input_size]
        :param query_r: query_r[batch_size,input_size]
        :param hidden_state: if it is none,it will be allocated a zero tensor hidden state
        :return:
        """
        update_value, reset_value = self.gate(torch.cat([message, query_r, hidden_state], dim=1)).chunk(2, dim=1)
        hidden_candidate = self.hidden_trans(torch.cat([message, reset_value * hidden_state], dim=1))
        hidden_state = (1 - update_value) * hidden_state + update_value * hidden_candidate
        return hidden_state