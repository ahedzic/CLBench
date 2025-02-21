
import sys
sys.path.append("..") 

import pickle
import torch
import numpy as np
import argparse
import scipy.sparse as ssp
from meta_gnn_model import *
from utils import *
from meta_scoring import meta_score
import random
# from logger import Logger

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc



log_print		= get_logger('testrun', 'log', get_config_dir())
def read_data(data_name, dir_path, cold_perc, blind):
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
            given_edges = graph['given_edges']
            num_nodes = graph['node_count']
            positive_edges = train_pos
            negative_edges = train_neg

            if (len(train_pos) > 0) and (len(train_neg) > 0):
                graph_data = {}

                if (cold_perc == 0.0) or (len(given_edges) == 0):
                    adj = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), sparse_sizes=[num_nodes, num_nodes])
                    graph_data['given_edges'] = []
                else:
                    graph_data['given_edges'] = torch.transpose(torch.tensor(given_edges), 1, 0)
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)
                    edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)
                    edge_weight = torch.ones(edge_index.size(1))
                    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
                    
                graph_data['adj'] = adj
                graph_data['pos'] = torch.transpose(torch.tensor(positive_edges), 1, 0)
                graph_data['neg'] = torch.transpose(torch.tensor(negative_edges), 1, 0)
                graph_data['x'] = graph['gnn_feature']
                graph_data['node_count'] = graph['node_count']
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
        

def train(model, score_func, graph, optimizer, device, iterations):
    expected_pos = graph['pos'].to(device)
    expected_neg = graph['neg'].to(device)
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    total_loss = total_examples = 0

    optimizer.zero_grad()

    given_edges = graph['given_edges']
    num_nodes = graph['node_count']

    for _ in range(iterations):
        h = model(x, adj)
        edge = expected_pos
        pos_out = score_func(x, h, edge[0], edge[1])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        if _ != (iterations - 1):
            total_edges_added = 0
            i_edge = []
            j_edge = []

            for i in range(len(pos_out)):
                if pos_out[i] >= 0.5:
                    i_edge.append(edge[0][i])
                    j_edge.append(edge[1][i])
                    total_edges_added += 1

            new_edges = torch.stack([torch.tensor(i_edge), torch.tensor(j_edge)]).to(device)
            new_edges_mask = torch.cat((new_edges, new_edges[[1,0]]), dim=1).to(torch.long).to(device)

            if len(given_edges):
                train_edge_mask = torch.cat((given_edges, given_edges[[1,0]]), dim=1).to(device)
                new_total_edges = torch.cat([train_edge_mask, new_edges_mask], 1).to(torch.long).to(device)
                new_edge_weight_mask = torch.ones(train_edge_mask.size(1) + new_edges_mask.size(1)).to(torch.float).to(device)
                adj = SparseTensor.from_edge_index(new_total_edges, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)
            else:
                new_edge_weight_mask = torch.ones(new_edges_mask.size(1)).to(torch.float).to(device)
                adj = SparseTensor.from_edge_index(new_edges_mask, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)

        edge = expected_neg
        neg_out = score_func(x, h, edge[0], edge[1])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)

        total_loss += loss.item() * num_examples
        total_examples += num_examples


    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, graph, edges, model, device, iterations):
    pos_preds = []
    x = graph['x'].to(device)
    adj = graph['adj'].to(device)
    given_edges = graph['given_edges']
    num_nodes = graph['node_count']

    for _ in range(iterations):
        h = model(x, adj)
        edges = edges
        pos_scores = score_func(x, h, edges[0], edges[1]).cpu()

        if _ != (iterations - 1):
            total_edges_added = 0
            i_edge = []
            j_edge = []

            for i in range(len(pos_scores)):
                if pos_scores[i] >= 0.5:
                    i_edge.append(edges[0][i])
                    j_edge.append(edges[1][i])
                    total_edges_added += 1

            new_edges = torch.stack([torch.tensor(i_edge), torch.tensor(j_edge)]).to(device)
            new_edges_mask = torch.cat((new_edges, new_edges[[1,0]]), dim=1).to(torch.long).to(device)

            if len(given_edges):
                train_edge_mask = torch.cat((given_edges, given_edges[[1,0]]), dim=1).to(device)
                new_total_edges = torch.cat([train_edge_mask, new_edges_mask], 1).to(torch.long).to(device)
                new_edge_weight_mask = torch.ones(train_edge_mask.size(1) + new_edges_mask.size(1)).to(torch.float).to(device)
                adj = SparseTensor.from_edge_index(new_total_edges, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)
            else:
                new_edge_weight_mask = torch.ones(new_edges_mask.size(1)).to(torch.float).to(device)
                adj = SparseTensor.from_edge_index(new_edges_mask, new_edge_weight_mask, [num_nodes, num_nodes]).to(device)
    
    pos_preds += [pos_scores]
          
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds


@torch.no_grad()
def test(model, score_func, data, evaluator_hit, evaluator_mrr, device, iterations):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred = test_edge(score_func, graph, graph['pos'], model, device, iterations)
        neg_pred = test_edge(score_func, graph, graph['neg'], model, device, iterations)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred = test_edge(score_func, graph, graph['pos'], model, device, iterations)
        neg_pred = test_edge(score_func, graph, graph['neg'], model, device, iterations)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred = test_edge(score_func, graph, graph['pos'], model, device, iterations)
        neg_pred = test_edge(score_func, graph, graph['neg'], model, device, iterations)
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
    parser.add_argument('--data_name', type=str, default='reddit')
    parser.add_argument('--input_size', type=int, default=602)
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='meta_score')
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')
    parser.add_argument('--max_nodes', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=3)

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--meta_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--filename', type=str, default='samples.npy')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ###### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    
    # state = torch.load('output_test/lr0.01_drop0.1_l20.0001_numlayer1_numPredlay2_numGinMlplayer2_dim64_best_run_0')

    #### 
    parser.add_argument('--eval_mrr_data_name', type=str, default='ogbl-citation2')

    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind)

    input_channel = args.input_size
    model = eval(args.gnn_model)(input_channel, args.meta_channels,
                    args.meta_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, args.max_nodes, args.cat_node_feat_mf).to(device)
    score_func = eval(args.score_model)(args.input_size, args.meta_channels, args.meta_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
    
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs),
       
    }

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
        for epoch in range(1, 1 + args.epochs):
            model.train()
            score_func.train()
            loss = 0.0
            loss_count = 0
            
            for graph in data['train']:
                loss += train(model, score_func, graph, optimizer, device, args.iterations)
                loss_count +=1
            
            if epoch % args.eval_steps == 0:
                model.eval()
                score_func.eval()
                results_rank, score_emb = test(model, score_func, data, evaluator_hit, evaluator_mrr, device, args.iterations)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        train_hits, valid_hits, test_hits = result
                       

                        log_print.info(
                            f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {(loss / loss_count):.4f}, '
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
        
    
    # print(best_metric_valid_str +' ' +best_auc_valid_str)

    print(best_metric_valid_str)
    best_auc_metric = best_valid_mean_metric


    return best_valid_mean_metric, best_auc_metric, result_all_run



if __name__ == "__main__":
    main()
   