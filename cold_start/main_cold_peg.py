import sys
sys.path.append("..") 
import pickle
import os, torch, dgl
import argparse

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from baseline_models.PEG.Graph_embedding import DeepWalk
from baseline_models.PEG.utils import laplacian_positional_encoding
from baseline_models.PEG.model import Net
from torch.utils.data import DataLoader
import random
import networkx as nx
import scipy.sparse as sp

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from utils import *
from torch_sparse import SparseTensor
import scipy.sparse as ssp

dir_path = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())


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

            if (len(train_pos) > 0) and (len(train_neg) > 0):
                given_edges = graph['given_edges']
                num_nodes = graph['node_count']
                positive_edges = train_pos
                negative_edges = train_neg

                if (cold_perc == 0.0) or (len(given_edges) == 0):
                    adj = SparseTensor(row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), sparse_sizes=[num_nodes, num_nodes])
                    edge_index = torch.tensor([])
                else:
                    adj_edge = torch.transpose(torch.tensor(given_edges), 1, 0)
                    edge_index = torch.cat((adj_edge,  adj_edge[[1,0]]), dim=1)
                    edge_weight = torch.ones(edge_index.size(1))
                    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

                graph_data = {}
                graph_data['adj'] = adj
                graph_data['pos'] = torch.tensor(positive_edges)
                graph_data['neg'] = torch.tensor(negative_edges)
                graph_data['x'] = graph['gnn_feature']
                graph_data['node_count'] = graph['node_count']
                graph_data['edge_index'] = edge_index
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

def train(model, optimizer, x, graph, device):
    m = torch.nn.Sigmoid()
    expected_pos = graph['pos'].to(device)
    expected_neg = graph['neg'].to(device)
    optimizer.zero_grad()

    edge_index = graph['edge_index'].to(device)

    edge = expected_pos.t()

    print("x shape", x.shape)
    print("edge_index shape", edge_index.shape)
    h = model.get_emb(x, edge_index)
    output = model.score(h, edge)
    pos_out = m(output)
    pos_out = torch.squeeze(pos_out)
    pos_loss = -torch.log(pos_out + 1e-15).mean()

    edge = expected_neg.t()
        
    output = model.score(h, edge)
    neg_out = m(output)
    neg_out = torch.squeeze(neg_out)
    neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

    loss = pos_loss + neg_loss
    loss.backward()
        
    optimizer.step()
    with torch.no_grad():
        model.fc.weight[0][0].clamp_(1e-5,100)

    return loss.item()

@torch.no_grad()
def test_edge(graph, input_data, x, model, device):
    preds = []
    edge_index = graph['edge_index']
    edge_index = edge_index.to(device)
    h = model.get_emb(x, edge_index)
    input_data = input_data.to(device)
    m = torch.nn.Sigmoid()
    edge = input_data.t()
    score = model.score(h, edge).cpu()
    cur_scores = m(score)
    preds += [cur_scores]
    
    if len(cur_scores.size()) > 0:
        pred_all = torch.cat(preds, dim=0)
    else:
        pred_all = torch.tensor(preds)

    return pred_all

torch.no_grad()
def test(model, data, x, evaluator_hit, evaluator_mrr, device):
    train_results = []
    valid_results = []
    test_results = []

    for graph in data['train_valid']:
        pos_pred = test_edge(graph, graph['pos'], x, model, device)
        neg_pred = test_edge(graph, graph['neg'], x, model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        k_list = [1, 3, 10, 100]
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        train_results.append((hits, mrr))

    for graph in data['valid']:
        pos_pred = test_edge(graph, graph['pos'], x, model, device)
        neg_pred = test_edge(graph, graph['neg'], x, model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        valid_results.append((hits, mrr))

    for graph in data['test']:  
        pos_pred = test_edge(graph, graph['pos'], x, model, device)
        neg_pred = test_edge(graph, graph['neg'], x, model, device)
        pos_pred = torch.flatten(pos_pred)
        neg_pred = torch.flatten(neg_pred)
        hits = evaluate_hits(evaluator_hit, pos_pred, neg_pred, k_list)
        mrr = evaluate_mrr(evaluator_mrr, pos_pred, neg_pred.repeat(pos_pred.size(0), 1))
        test_results.append((hits, mrr))
    
    result = get_average_results(train_results, valid_results, test_results)
    
    score_emb = [pos_pred.cpu(),neg_pred.cpu(), pos_pred.cpu(), neg_pred.cpu()]

    return result, score_emb

def get_embeddings(graph, PE_method, PE_dim, device):
    adj = graph['adj'].to_dense()
    train_matrix=np.copy(adj)
    features = graph['x']
        
    if PE_method == 'DW':
        #deepwalk
        G = nx.DiGraph(train_matrix)
        model_emb = DeepWalk(G,walk_length=80, num_walks=10,workers=1)#init model
        model_emb.train(embed_size = PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        embeddings = []
        for i in range(len(emb)):
            embeddings.append(emb[i])
        embeddings = np.array(embeddings)

    elif PE_method == 'LE':
        #LAP
        sp_adj = sp.coo_matrix(train_matrix)
        g = dgl.from_scipy(sp_adj)
        #g = dgl.add_self_loop(g)
        embeddings = np.array(laplacian_positional_encoding(g, PE_dim))
        embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)

    x = torch.cat((torch.tensor(embeddings), features), 1)
    # edge_index = np.array(train_edge_index).transpose()
    # edge_index = torch.from_numpy(edge_index)
    
    x = x.to(device)
    
    return x, len(features[1])


def main():
    parser = argparse.ArgumentParser(description='homo')

    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--input_size', type=int, default=602)
    parser.add_argument('--cold_perc', type=float, default=0.25)
    parser.add_argument('--blind', type=str, default='edge')
    parser.add_argument('--PE_dim', type=int, default=128, help = 'dimension of positional encoding')
    parser.add_argument('--hidden_dim', type=int, default=256, help = 'hidden dimension')
    parser.add_argument('--input_dir', type=str, default=os.path.join(get_root_dir(), "dataset"))
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--feature_type', type=str, default="N", help = 'features type, N means node feature, C means constant feature (node degree)',
                    choices = ['N', 'C'])
    parser.add_argument('--PE_method', type=str, default="LE", help = 'positional encoding techniques',
                    choices = ['DW', 'LE'])
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--save', action='store_true', default=False)

    ### debug
    # parser.add_argument('--device', type=int, default=2)
    # parser.add_argument('--random_partition', action='store_true', default=False,help = 'whether to use random partition while training')
    parser.add_argument('--no_pe', action='store_true', help = 'whether to use pe')

    args = parser.parse_args()
    
    print(args)

    # [115, 105, 100]
    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    device = torch.device('cpu')#args.device

    data = read_data(args.data_name, args.input_dir, args.cold_perc, args.blind)

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

   

    

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)
        
        init_seed(seed)

        # edge_index = edge_index.cuda(device)
        model = Net(in_feats_dim = args.input_size, pos_dim = args.PE_dim, hidden_dim = args.input_size, no_pe=args.no_pe)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters() ,lr=args.lr, weight_decay=args.l2)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_l2'+ str(args.l2)  +'_dim'+str(args.hidden_dim) + '_'+ 'best_run_'+str(seed)

        best_valid = 0
        kill_cnt = 0

        small_epoch_list = []
        for i in range(2):
            small_epoch_list.append(i)

        for epoch in range(1, 1 + args.epochs):
            model.train()
            loss = 0.0
            loss_count = 0

            for graph in data['train']:
                x, feat_length = get_embeddings(data['train'][0], args.PE_method, args.PE_dim, device)
                loss += train(model, optimizer, x, graph, device)
                loss_count +=1

            if epoch % args.eval_steps == 0:
                model.eval()
                results_rank, score_emb = test(model, data, x, evaluator_hit, evaluator_mrr, device)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

            if epoch % args.eval_steps == 0 :
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
        
    
    print(best_metric_valid_str)

    return best_valid_mean_metric, best_auc_metric, result_all_run


if __name__ == "__main__":
    main()
   