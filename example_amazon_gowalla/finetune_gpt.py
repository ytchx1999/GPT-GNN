import sys
from GPT_GNN.data import *
from GPT_GNN.model import *
from warnings import filterwarnings
from GPT_GNN.model import LinkPredictor
from GPT_GNN.data import args_print, load_gnn
from torch_geometric.loader import NeighborSampler

from sklearn.metrics import f1_score
filterwarnings("ignore")

import argparse
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description='Fine-Tuning on Reddit classification task')

'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='./dataset',
                    help='The address of preprocessed graph.')
parser.add_argument('--use_pretrain', help='Whether to use pre-trained model', action='store_true', default=True)
parser.add_argument('--pretrain_model_dir', type=str, default='./models/gpt_all_cs',
                    help='The address for pretrained model.')
parser.add_argument('--model_dir', type=str, default='./models/gpt_all_reddit',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--task_name', type=str, default='reddit',
                    help='The name of the stored models and optimization results.')
parser.add_argument('--cuda', type=int, default=1,
                    help='Avaiable GPU ID')     
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=128,
                    help='How many nodes to be sampled per layer per type')
'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')


'''
    Optimization arguments
'''
parser.add_argument('--optimizer', type=str, default='adamw',
                    choices=['adamw', 'adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--scheduler', type=str, default='cosine',
                    help='Name of learning rate scheduler.' , choices=['cycle', 'cosine'])
parser.add_argument('--data_percentage', type=int, default=0.1,
                    help='Percentage of training and validation data to use')
parser.add_argument('--n_epoch', type=int, default=40,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')    
parser.add_argument('--n_batch', type=int, default=16,
                    help='Number of batch (sampled graphs) for each epoch') 
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of output nodes for training')    
parser.add_argument('--clip', type=int, default=0.5,
                    help='Gradient Norm Clipping') 
parser.add_argument('--data_type', type=str, default="amazon", help='Type of dataset')
parser.add_argument('--task_type', type=str, default="time_trans", help='Type of task')
parser.add_argument('-d', '--data', type=str, help='Dataset name', default='amazon_beauty')

args = parser.parse_args()
args_print(args)

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

if args.task_type in ['field_trans', 'tf_trans']:
    if args.data_type == 'amazon':
        data_name = 'amazon_acs'
    elif args.data_type == 'gowalla':
        data_name = 'gowalla_Food'
else:
    data_name = args.data

if not os.path.exists('./models'):
    os.makedirs('./models', exist_ok=True)

graph = dill.load(open('./dataset/graph_downstream_{}_{}.pk'.format(args.data, args.task_type), 'rb'))

val_test_sampler = dill.load(open('./dataset/val_test_sampler_downstream_{}_{}.pk'.format(args.data, args.task_type), 'rb'))
val_rand_sampler = val_test_sampler['val_rand_sampler']
nn_val_rand_sampler = val_test_sampler['nn_val_rand_sampler']
test_rand_sampler = val_test_sampler['test_rand_sampler']
nn_test_rand_sampler = val_test_sampler['nn_test_rand_sampler']

val_test_data = dill.load(open('./dataset/val_test_data_downstream_{}_{}.pk'.format(args.data, args.task_type), 'rb'))
val_data = val_test_data['val_data']
new_node_val_data = val_test_data['new_node_val_data']
test_data = val_test_data['test_data']
new_node_test_data = val_test_data['new_node_test_data']

train_data_np = dill.load(open('./dataset/train_data_downstream_{}_{}.pk'.format(args.data, args.task_type), 'rb'))

target_type = 'def'
train_target_nodes = graph.train_target_nodes
valid_target_nodes = graph.valid_target_nodes
test_target_nodes  = graph.test_target_nodes

types = graph.get_types()
# criterion = nn.NLLLoss()
criterion = nn.BCEWithLogitsLoss()

max_node_idx = max(train_target_nodes.max(), max(valid_target_nodes.max(), test_target_nodes.max()))
train_adj = [[] for _ in range(max_node_idx + 1)]
for source, destination in zip(train_data_np.sources, train_data_np.destinations):
    train_adj[source].append((destination))
    train_adj[destination].append((source))

def node_classification_sample(seed, nodes, time_range, stage='train'):
    '''
        sub-graph sampling and label preparation for node classification:
        (1) Sample batch_size number of output nodes (papers) and their time.
    '''
    np.random.seed(seed)
    if stage == 'train':
        samp_nodes = np.random.choice(nodes, args.batch_size, replace = False)
        pos_nodes = []
        neg_nodes = []
        for node in samp_nodes:
            pos = np.random.choice(np.array(train_adj[int(node)]), 1, replace = False)[0]
            neg = np.random.choice(nodes, 1, replace = False)[0]
            pos_nodes.append(pos)
            neg_nodes.append(neg)
        pos_nodes = np.array(pos_nodes)
        neg_nodes = np.array(neg_nodes)

        node_map = {}
        all_nodes = np.concatenate([samp_nodes, pos_nodes, neg_nodes])
        all_nodes_set = set(all_nodes.tolist())
        all_nodes_list = list(all_nodes_set)
        for idx, nd in enumerate(all_nodes_list):
            node_map[nd] = idx
        
        x_ids = []
        pos_x_ids = []
        neg_x_ids = []
        for nd in samp_nodes:
            x_ids.append(node_map[nd])
        for nd in pos_nodes:
            pos_x_ids.append(node_map[nd])
        for nd in neg_nodes:
            neg_x_ids.append(node_map[nd])  
        
        x_ids = np.array(x_ids)
        pos_x_ids = np.array(pos_x_ids)
        neg_x_ids = np.array(neg_x_ids)

        feature, times, edge_list, _, texts = sample_subgraph(graph, time_range, \
                    inp = {target_type: np.concatenate([np.array(all_nodes_list), np.ones(len(all_nodes_list))]).reshape(2, -1).transpose()}, \
                    sampled_depth = args.sample_depth, sampled_number = args.sample_width, feature_extractor = feature_reddit)
        
        node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
                to_torch(feature, times, edge_list, graph)
    else:
        samp_nodes = nodes
        if stage == 'valid':
            val_src, val_dst = val_data.sources, val_data.destinations
            _, val_neg = val_rand_sampler.sample(len(val_src))
            # nn_val_src, nn_val_dst = new_node_val_data.sources, new_node_val_data.destinations
            # _, nn_val_neg = nn_val_rand_sampler.sample(len(nn_val_src))

            node_map = {}
            all_nodes = np.concatenate([val_src, val_dst, val_neg])
            all_nodes_set = set(all_nodes.tolist())
            all_nodes_list = list(all_nodes_set)
            for idx, nd in enumerate(all_nodes_list):
                node_map[nd] = idx
            
            x_ids = []
            pos_x_ids = []
            neg_x_ids = []
            for nd in val_src:
                x_ids.append(node_map[nd])
            for nd in val_dst:
                pos_x_ids.append(node_map[nd])
            for nd in val_neg:
                neg_x_ids.append(node_map[nd])  
            
            x_ids = np.array(x_ids)
            pos_x_ids = np.array(pos_x_ids)
            neg_x_ids = np.array(neg_x_ids)
        
            feature, times, edge_list, _, texts = sample_subgraph(graph, time_range, \
                    inp = {target_type: np.concatenate([val_src, val_dst, val_neg, np.ones(val_src.shape[0] * 3)]).reshape(2, -1).transpose()}, \
                    sampled_depth = args.sample_depth, sampled_number = args.sample_width, feature_extractor = feature_reddit)
        
            node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
                    to_torch(feature, times, edge_list, graph)
        elif stage == 'test':
            test_src, test_dst = test_data.sources, test_data.destinations
            _, test_neg = test_rand_sampler.sample(len(test_src))
            # nn_test_src, nn_test_dst = new_node_test_data.sources, new_node_test_data.destinations
            # _, nn_test_neg = nn_test_rand_sampler.sample(len(nn_test_src))

            node_map = {}
            all_nodes = np.concatenate([test_src, test_dst, test_neg])
            all_nodes_set = set(all_nodes.tolist())
            all_nodes_list = list(all_nodes_set)
            for idx, nd in enumerate(all_nodes_list):
                node_map[nd] = idx
            
            x_ids = []
            pos_x_ids = []
            neg_x_ids = []
            for nd in test_src:
                x_ids.append(node_map[nd])
            for nd in test_dst:
                pos_x_ids.append(node_map[nd])
            for nd in test_neg:
                neg_x_ids.append(node_map[nd])  
            
            x_ids = np.array(x_ids)
            pos_x_ids = np.array(pos_x_ids)
            neg_x_ids = np.array(neg_x_ids)

            feature, times, edge_list, _, texts = sample_subgraph(graph, time_range, \
                    inp = {target_type: np.concatenate([test_src, test_dst, test_neg, np.ones(test_src.shape[0] * 3)]).reshape(2, -1).transpose()}, \
                    sampled_depth = args.sample_depth, sampled_number = args.sample_width, feature_extractor = feature_reddit)
        
            node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
                    to_torch(feature, times, edge_list, graph)
    
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, pos_x_ids, neg_x_ids
    
    
def prepare_data(pool):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    for batch_id in np.arange(args.n_batch):
        p = pool.apply_async(node_classification_sample, args=(randint(), train_target_nodes, {1: True}, 'train'))
        jobs.append(p)
    p = pool.apply_async(node_classification_sample, args=(randint(), valid_target_nodes, {1: True}, 'valid'))
    jobs.append(p)
    return jobs

stats = []
res = []
best_val   = 0
best_val_res = {}
train_step = 0

pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)


'''
    Initialize GNN (model is specified by conv_name) and Classifier
'''
gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature[target_type]['emb'].values[0]), n_hid = args.n_hid, \
          n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout, num_types = len(types), \
          num_relations = len(graph.get_meta_graph()) + 1, prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = False)
if args.use_pretrain:
    gnn.load_state_dict(load_gnn(torch.load('./models/gpt_pretrain_{}_{}.pk'.format(data_name, args.task_type))), strict = False)
    print('Load Pre-trained Model from ./models/gpt_pretrain_{}_{}.pk'.format(data_name, args.task_type))
# classifier = Classifier(args.n_hid, graph.y.max().item() + 1)
link_predictor = LinkPredictor(args.n_hid)

# model = nn.Sequential(gnn, classifier).to(device)
model = nn.Sequential(gnn, link_predictor).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4)


if args.scheduler == 'cycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.02, anneal_strategy='linear', final_div_factor=100,\
                        max_lr = args.max_lr, total_steps = args.n_batch * args.n_epoch + 1)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)


for epoch in np.arange(args.n_epoch) + 1:
    '''
        Prepare Training and Validation Data
    '''
    train_data = [job.get() for job in jobs[:-1]]
    valid_data = jobs[-1].get()
    pool.close()
    pool.join()
    '''
        After the data is collected, close the pool and then reopen it.
    '''
    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)
    et = time.time()
    print('Data Preparation: %.1fs' % (et - st))
    
    '''
        Train
    '''
    model.train()
    train_losses = []
    for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, pos_x_ids, neg_x_ids in train_data:
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                               edge_time.to(device), edge_index.to(device), edge_type.to(device))
        res = node_rep[x_ids]
        pos = node_rep[pos_x_ids]
        neg = node_rep[neg_x_ids]
        pos_score = link_predictor(res, pos)
        neg_score = link_predictor(res, neg)
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        score = torch.cat([pos_score, neg_score]).squeeze(-1)
        labels = torch.cat([pos_label, neg_label]).squeeze(-1)

        loss = criterion(score, labels.to(device))

        optimizer.zero_grad() 
        torch.cuda.empty_cache()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
        train_step += 1
        scheduler.step(train_step)
        del res, loss
    '''
        Valid
    '''
    model.eval()
    with torch.no_grad():
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, pos_x_ids, neg_x_ids = valid_data
        node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                                   edge_time.to(device), edge_index.to(device), edge_type.to(device))
        
        res = node_rep[x_ids]
        pos = node_rep[pos_x_ids]
        neg = node_rep[neg_x_ids]
        pos_score = link_predictor(res, pos)
        neg_score = link_predictor(res, neg)

        pred_score = np.concatenate([(pos_score.sigmoid()).cpu().detach().numpy(), (neg_score.sigmoid()).cpu().detach().numpy()])
        true_label = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])

        val_ap = average_precision_score(true_label, pred_score)
        val_auc = roc_auc_score(true_label, pred_score)
        val_f1_micro = f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='micro')
        val_f1_macro = f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='macro')
        
        '''
            Calculate Valid F1. Update the best model based on highest F1 score.
        '''
        # valid_f1 = f1_score(res.argmax(dim=1).cpu().tolist(), ylabel.tolist(), average='micro')
        
        if val_ap > best_val:
            best_val = val_ap
            best_val_res['ap'] = val_ap
            best_val_res['auc'] = val_auc
            best_val_res['f1_micro'] = val_f1_micro
            best_val_res['f1_macro'] = val_f1_macro
            torch.save(model, './models/gpt_downstream_{}_{}.pk'.format(args.data, args.task_type))
            print('UPDATE!!!')
        
        st = time.time()
        print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid ap: %.4f Valid auc: %.4f Valid f1_micro: %.4f Valid f1_macro: %.4f") % \
              (epoch, (st-et), optimizer.param_groups[0]['lr'], np.average(train_losses), val_ap, val_auc, val_f1_micro, val_f1_macro))
        del res
    del train_data, valid_data

print('-'*50)
print(("Final Valid ap: %.4f Valid auc: %.4f Valid f1_micro: %.4f Valid f1_macro: %.4f") % \
    (best_val_res['ap'], best_val_res['auc'], best_val_res['f1_micro'], best_val_res['f1_macro']))
print('-'*50)

best_model = torch.load('./models/gpt_downstream_{}_{}.pk'.format(args.data, args.task_type)).to(device)
best_model.eval()
gnn, classifier = best_model
with torch.no_grad():
    test_res_ap = []
    test_res_auc = []
    test_res_f1_micro = []
    test_res_f1_macro = []
    for _ in range(10):
        node_feature, node_type, edge_time, edge_index, edge_type, x_ids, pos_x_ids, neg_x_ids = \
                    node_classification_sample(randint(), test_target_nodes, {1: True}, 'test')
        paper_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
                    edge_time.to(device), edge_index.to(device), edge_type.to(device))

        res = paper_rep[x_ids]
        pos = paper_rep[pos_x_ids]
        neg = paper_rep[neg_x_ids]
        pos_score = link_predictor(res, pos)
        neg_score = link_predictor(res, neg)

        pred_score = np.concatenate([(pos_score.sigmoid()).cpu().detach().numpy(), (neg_score.sigmoid()).cpu().detach().numpy()])
        true_label = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])

        test_ap = average_precision_score(true_label, pred_score)
        test_auc = roc_auc_score(true_label, pred_score)
        test_f1_micro = f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='micro')
        test_f1_macro = f1_score(true_label, np.where(pred_score > 0.5, 1, 0), average='macro')

        test_res_ap += [test_ap]
        test_res_auc += [test_auc]
        test_res_f1_micro += [test_f1_micro]
        test_res_f1_macro += [test_f1_macro]
    
    test_res_ap = np.array(test_res_ap)
    test_res_auc = np.array(test_res_auc)
    test_res_f1_micro = np.array(test_res_f1_micro)
    test_res_f1_macro = np.array(test_res_f1_macro)
    # print('Best Test F1: %.4f' % np.average(test_res))
    print(f'Final test ap: {np.mean(test_res_ap)} ± {np.std(test_res_ap)}', flush=True)
    print(f'Final test auc: {np.mean(test_res_auc)} ± {np.std(test_res_auc)}', flush=True)
    print(f'Final test f1_micro: {np.mean(test_res_f1_micro)} ± {np.std(test_res_f1_micro)}', flush=True)
    print(f'Final test f1_macro: {np.mean(test_res_f1_macro)} ± {np.std(test_res_f1_macro)}', flush=True)
