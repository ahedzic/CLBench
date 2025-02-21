import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
from scipy.sparse import csr_matrix

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]

        hs = hidden[sub]
        hr = self.rela_embed(rel) 

        r_idx = edges[:,0]
        h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs  + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new

class RED_GNN(torch.nn.Module):
    def __init__(self, n_layer, hidden_dim, attn_dim, n_rel, dropout):
        super(RED_GNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.n_rel = n_rel
        #acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = nn.ReLU()#acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
       
        self.dropout = nn.Dropout(dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

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

    def forward(self, subs, rels, n_ent, KG, given_subs, device):#, mode='transductive'):
        n = len(subs)

        #n_ent = self.loader.n_ent if mode=='transductive' else self.loader.n_ent_ind

        q_sub = subs.to(device)
        q_rel = rels.to(device)

        h0 = torch.zeros((1, n, self.hidden_dim)).to(device)
        nodes = torch.cat([torch.arange(n).unsqueeze(1).to(device), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).to(device)

        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.get_neighbors(nodes.data.cpu().numpy(), KG, given_subs, n_ent)
    
            hidden = self.gnn_layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).to(device).index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1) 
        scores_all = torch.zeros((n, n_ent)).to(device)
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        
        return scores_all