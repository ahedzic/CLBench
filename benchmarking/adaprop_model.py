import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.W_attn = nn.Linear(attn_dim, 1, bias=False)
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
        mess1 = hs 
        mess2 = mess1 + hr
        alpha_2 = torch.sigmoid(self.W_attn(nn.ReLU()(self.Ws_attn(mess1) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = mess2*alpha_2
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        
        hidden_new =  self.W_h(message_agg)
        hidden_new = self.act(hidden_new)
        
        return hidden_new

class AdaProp(torch.nn.Module):
    def __init__(self, n_layer, hidden_dim, attn_dim, n_rel, dropout):
        super(AdaProp, self).__init__()
        self.n_layer = n_layer
        self.init_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.n_rel = n_rel
        self.increase = True
        self.topk = 100
        #acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = nn.ReLU()#acts[params.act]

        self.layers = []
        self.Ws_layers = []
        for i in range(self.n_layer):
            self.layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
            self.Ws_layers.append(nn.Linear(self.hidden_dim, 1, bias=False))
        self.layers = nn.ModuleList(self.layers)
        self.Ws_layers = nn.ModuleList(self.Ws_layers)
       
        self.dropout = nn.Dropout(dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)         # get score
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)

    def soft_to_hard(self, i, hidden, nodes, n_ent, batch_size, old_nodes_new_idx):
        n_node = len(nodes)
        bool_diff_node_idx = torch.ones(n_node).bool().cuda()
        bool_diff_node_idx[old_nodes_new_idx] = False
        bool_same_node_idx = ~bool_diff_node_idx
        diff_nodes = nodes[bool_diff_node_idx]
        diff_node_logits = self.Ws_layers[i](hidden[bool_diff_node_idx].detach()).squeeze(-1)

        soft_all = torch.ones((batch_size, n_ent)) * float('-inf')
        soft_all = soft_all.cuda()
        soft_all[diff_nodes[:,0], diff_nodes[:,1]] = diff_node_logits
        soft_all = F.softmax(soft_all, dim=-1)

        diff_node_logits = self.topk * soft_all[diff_nodes[:,0], diff_nodes[:,1]]
        _, argtopk = torch.topk(soft_all, k=self.topk, dim=-1)
        r_idx = torch.arange(batch_size).unsqueeze(1).repeat(1,self.topk).cuda()
        hard_all = torch.zeros((batch_size, n_ent)).bool().cuda()
        hard_all[r_idx,argtopk] = True
        bool_sampled_diff_nodes = hard_all[diff_nodes[:,0], diff_nodes[:,1]]
        
        hidden[bool_diff_node_idx][bool_sampled_diff_nodes] *= (1 - diff_node_logits[bool_sampled_diff_nodes].detach() + diff_node_logits[bool_sampled_diff_nodes]).unsqueeze(1)
        bool_same_node_idx[bool_diff_node_idx] = bool_sampled_diff_nodes

        return hidden, bool_same_node_idx

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
        #n_ent = self.loader.n_ent if mode=='transductive' else self.loader.n_ent_ind
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
        h0 = torch.zeros((1, n,self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()
        
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.get_neighbors(nodes.data.cpu().numpy(), KG, given_subs, n_ent)
            hidden = self.layers[i](q_sub, q_rel, hidden, edges, nodes.size(0), old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gru(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            
            if i < self.n_layer-1:
                if self.increase:
                    hidden, bool_same_nodes = self.soft_to_hard(i, hidden, nodes, n_ent, n, old_nodes_new_idx)
                else:
                    exit()
                    
                nodes = nodes[bool_same_nodes]
                hidden = hidden[bool_same_nodes]
                h0 = h0[:,bool_same_nodes]
                
        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, n_ent)).cuda()
        scores_all[[nodes[:,0], nodes[:,1]]] = scores

        return scores_all