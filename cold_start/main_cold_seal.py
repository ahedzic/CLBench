    
import sys
sys.path.append("..") 

import torch
import numpy as np
import pickle
import argparse
import scipy.sparse as ssp
from baseline_models.seal_utils import *
from gnn_model import *
from utils import *
from scoring import mlp_score
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from baseline_models.seal_dataset import SEALDataset, SEALDynamicDataset
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

from baseline_models.seal_utils import *
import time
from gnn_model import *
from torch.nn import BCEWithLogitsLoss

dir_path = get_root_dir()
log_print		= get_logger('testrun', 'log', get_config_dir())



from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
import scipy.sparse as ssp
from baseline_models.seal_utils import *
from torch_sparse import coalesce


    
class homo_data(torch.nn.Module):
    def __init__(self, edge_index, num_nodes, x=None, edge_weight=None):
        super(homo_data).__init__()
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        if x != None: self.x = x
        else: self.x = None


        if edge_weight != None: self.edge_weight = edge_weight
        else: self.edge_weight = None
    
    
# sys.modules[__name__] = homo_data()



def read_data(data_name, dir_path, cold_perc, blind, args):
    if cold_perc > 0.0:
        if cold_perc == 0.25:
            cold_part = '25'
        if cold_perc == 0.75:
            cold_part = '75'

        path = dir_path+ '/{}/{}_{}_graphs.pkl'.format(data_name, data_name, cold_part+'_'+blind)
    else:
        path = dir_path+ '/{}/{}_{}_graphs.pkl'.format(data_name, data_name, 'true')
    graphs_input = open(path, 'rb')
    graphs = pickle.load(graphs_input)
    data = {
        'train': [],
        'valid': [],
        'test': []
    }
    
    for graphs_key in graphs.keys():
        for graph in graphs[graphs_key]:
            train_pos = graph['pos_edges']
            train_neg = graph['neg_edges']

            if (len(train_pos) > 0) and (len(train_neg) > 0):
                given_edges = graph['given_edges']
                num_nodes = graph['node_count']
                positive_edges = train_pos
                negative_edges = train_neg

                if (cold_perc == 0.0) or (len(given_edges) == 0):
                    adj = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), sparse_sizes=[num_nodes, num_nodes])
                    A = ssp.csr_matrix(torch.zeros((num_nodes, num_nodes)), shape=(num_nodes, num_nodes))
                else:
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)
                    edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)
                    edge_weight = torch.ones(edge_index.size(1))
                    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

                    edge_index = adj_edge.cpu()
                    edge_weight = torch.ones(edge_index.size(1), dtype=float)
                    A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), 
                                        shape=(num_nodes, num_nodes))
                    
                train_pos = torch.tensor(train_pos)
                train_neg = torch.tensor(train_neg)
                links = torch.cat([train_pos, train_neg]).t().tolist()
                labels = [1] * train_pos.size(1) + [0] * train_neg.size(1)
                src, dst = links[0], links[1]
                print(src)
                y = labels
                tmp = k_hop_subgraph(src, dst, args.num_hops, A, args.ratio_per_hop, 
                                    args.max_nodes_per_hop, node_features=graph['gnn_feature'], 
                                    y=y, directed=False, A_csc=A.tocsc())
                data = construct_pyg_graph(*tmp, 'drnl')
                        
                graph_data = {}
                graph_data['adj'] = adj
                graph_data['A'] = A
                graph_data['pos'] = torch.tensor(positive_edges)
                graph_data['neg'] = torch.tensor(negative_edges)
                graph_data['x'] = graph['gnn_feature']
                graph_data['node_count'] = graph['node_count']
                graph_data['data'] = data
                data[graphs_key].append(graph_data)

    train_valid_count = len(data['valid'])
    data['train_valid'] = data['train'][:train_valid_count]

    return data



def get_average_results(train, valid, test):
    all_result = {}
    train_total = 0.0
    valid_total = 0.0
    test_total = 0.0
    result_mrr_train = {'MRR': 0.0}
    result_mrr_valid = {'MRR': 0.0}
    result_mrr_test = {'MRR': 0.0}

    for K in [1,3,10, 100]:
        result_mrr_train[f'Hits@{K}'] = 0.0
        result_mrr_valid[f'Hits@{K}'] = 0.0
        result_mrr_test[f'Hits@{K}'] = 0.0

    for result in train:
        train_total += 1.0
        result_mrr_train['MRR'] += result[1]['MRR']

        for K in [1,3,10, 100]:
            result_mrr_train[f'Hits@{K}'] += result[0][f'Hits@{K}']

    for result in valid:
        valid_total += 1.0
        result_mrr_valid['MRR'] += result[1]['MRR']

        for K in [1,3,10, 100]:
            result_mrr_valid[f'Hits@{K}'] += result[0][f'Hits@{K}']

    for result in test:
        test_total += 1.0
        result_mrr_test['MRR'] += result[1]['MRR']

        for K in [1,3,10, 100]:
            result_mrr_test[f'Hits@{K}'] += result[0][f'Hits@{K}']

    result_mrr_train['MRR'] = result_mrr_train['MRR'] / train_total
    result_mrr_valid['MRR'] = result_mrr_valid['MRR'] / valid_total
    result_mrr_test['MRR'] = result_mrr_test['MRR'] / test_total

    for K in [1,3,10, 100]:
        result_mrr_train[f'Hits@{K}'] = result_mrr_train[f'Hits@{K}'] / train_total
        result_mrr_valid[f'Hits@{K}'] = result_mrr_valid[f'Hits@{K}'] / valid_total
        result_mrr_test[f'Hits@{K}'] = result_mrr_test[f'Hits@{K}'] / test_total

    all_result['MRR'] = (result_mrr_train['MRR'], result_mrr_valid['MRR'], result_mrr_test['MRR'])
    for K in [1,3,10, 100]:
        all_result[f'Hits@{K}'] = (result_mrr_train[f'Hits@{K}'], result_mrr_valid[f'Hits@{K}'], result_mrr_test[f'Hits@{K}'])
    
    return all_result  

def train(model, graph, optimizer, device, args):
    optimizer.zero_grad()
    data = graph['data']
    x = graph['x'].to(device)
    edge_weight = graph['edge_weight'] if args.use_edge_weight else None
    node_id = None
    logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
    loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test_edge(graph, model, device):
    y_pred, y_true = [], []
    data = graph['data'].to(device)
    x = data.x
    edge_weight = data.edge_weight
    node_id = None
    logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
    y_pred.append(logits.view(-1).cpu())
    y_true.append(data.y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]
  
    pos_preds = []
    neg_preds = []

    pos_preds += [pos_val_pred]
    neg_preds += [neg_val_pred]

    return pos_preds, neg_preds

@torch.no_grad()
def test(model, data, evaluator_hit, evaluator_mrr, device):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred, neg_pred = test_edge(graph, model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred, neg_pred = test_edge(graph, model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred, neg_pred = test_edge(graph, model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        test_results.append((hits, mrr))
    
    result = get_average_results(train_results, valid_results, test_results)
    
    score_emb = [pos_pred.cpu(),neg_pred.cpu(), pos_pred.cpu(), neg_pred.cpu()]

    return result, score_emb

def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='DGCNN')
    parser.add_argument('--score_model', type=str, default='mlp_score')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4, 
                    help="number of workers for dynamic mode; 0 if not dynamic")
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ####### seal 
    parser.add_argument('--dynamic_train', action='store_true', 
                    help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    
    parser.add_argument('--node_label', type=str, default='drnl',  help="which specific labeling trick to use")
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--num_hops', type=int, default=3)
    parser.add_argument('--save_appendix', type=str, default='', 
                    help="an appendix to the save directory")
    parser.add_argument('--data_appendix', type=str, default='', 
                    help="an appendix to the data directory")
    parser.add_argument('--use_feature', action='store_true', 
                    help="whether to use raw node features as GNN input")
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--use_edge_weight', action='store_true', 
                    help="whether to consider edge weight in GNN")

    # parser.add_argument('--num_hops', type=int, default=3)
    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind, args)

    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
    if args.data_appendix == '':
        args.data_appendix = '_h{}_{}_rph{}'.format(
        args.num_hops, args.node_label, ''.join(str(args.ratio_per_hop).split('.')))
    if args.max_nodes_per_hop is not None:
        args.data_appendix += '_mnph{}'.format(args.max_nodes_per_hop)

    if not args.dynamic_train and not args.dynamic_val and not args.dynamic_test:
        args.num_workers = 0

    max_z = 1000  # set a large max_z so that every z has embeddings to look up

    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
    }

    emb = None

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        init_seed(seed)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
                      None, args.dynamic_train, use_feature=args.use_feature, 
                      node_embedding=emb).to(device)
        parameters = list(model.parameters())


        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.l2)
        if args.gnn_model == 'DGCNN':
            print(f'SortPooling k is set to {model.k}')

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            model.train()
            loss = 0.0
            loss_count = 0

            for graph in data['train']:
                loss += train(model, graph, optimizer, device, args)
                loss_count +=1
            
            if epoch % args.eval_steps == 0:
                model.eval()
                results_rank, score_emb= test(model, data, evaluator_hit, evaluator_mrr, device)
               

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                        log_print.info(
                            
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0


                    if args.save:

                        save_emb(score_emb, save_path)
                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        print(key)
        
        best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()

        if key == eval_metric:
            best_metric_valid_str = best_metric
            best_valid_mean_metric = best_valid_mean


            
        if key == 'AUC':
            best_auc_valid_str = best_metric
            best_auc_metric = best_valid_mean

        result_all_run[key] = [mean_list, var_list]
        



    
    
    print(best_metric_valid_str +' ')



if __name__ == "__main__":

    main()

    