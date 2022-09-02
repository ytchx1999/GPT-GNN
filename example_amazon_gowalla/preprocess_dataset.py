from torch_geometric.datasets import Reddit
from GPT_GNN.data import *
from data_processing import compute_time_statistics, get_data_no_label
import os
import argparse
from utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Link Prediction')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='amazon_beauty')
    parser.add_argument('--bs', type=int, default=512, help='Batch_size')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate') #0.0001
    parser.add_argument('--drop', type=float, default=0.5, help='Dropout')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
    parser.add_argument('--model', type=str, default="graphsage", choices=["graphsage", "sgc", "gcn", "gin", "gat", "dgi"], help='Type of embedding module')
    parser.add_argument('--n_hidden', type=int, default=256, help='Dimensions of the hidden')
    parser.add_argument("--fanout", type=str, default='15,10,5', help='Neighbor sampling fanout')
    # parser.add_argument("--fanout_sgc", type=str, default='0', help='SGC neighbor sampling fanout')
    parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
    parser.add_argument('--data_type', type=str, default="amazon", help='Type of dataset')
    parser.add_argument('--task_type', type=str, default="time_trans", help='Type of task')
    parser.add_argument('--mode', type=str, default="downstream", help='pretrain or downstream')
    parser.add_argument('--seed', type=int, default=0, help='Seed for all')
    # parser.add_argument('--k_hop', type=int, default=3, help='K-hop for SGC')
    parser.add_argument('--learn_eps', action="store_true", help='learn the epsilon weighting')
    parser.add_argument('--aggr_type', type=str, default="mean", choices=["sum", "mean", "max"], help='type of neighboring pooling: sum, mean or max')
    parser.add_argument('--dgi_lam', type=float, default=1., help='coefficient of dgi loss')
    parser.add_argument('--base_dir', type=str, default="/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/chihuixuan/dgl-lp-baseline/", help='Base dir of dataset')

    args = parser.parse_args()
    set_seed(args.seed)

    n_feats, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data_no_label(args.data,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features, \
                              have_edge=False, base_dir=args.base_dir, data_type=args.data_type, task_type=args.task_type, mode=args.mode, seed=args.seed)
    
    train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

    # Initialize validation and test neighbor finder to retrieve temporal graph
    full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

    # Initialize negative samplers. Set seeds for validation and testing so negatives are the same
    # across different runs
    # NB: in the inductive setting, negatives are sampled only amongst other new nodes
    train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
    val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed)
    nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=args.seed)
    test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=args.seed)
    nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=args.seed)

    if not os.path.exists('./dataset'):
        os.makedirs('./dataset', exist_ok=True)
    
    if args.task_type in ['field_trans', 'tf_trans']:
        if args.data_type == 'amazon':
            data_name = 'amazon_acs'
        elif args.data_type == 'gowalla':
            data_name = 'gowalla_Food'
    else:
        data_name = args.data

    if args.mode == 'pretrain':
        edge_index = torch.from_numpy(np.concatenate((full_data.sources.reshape(-1, 1), full_data.destinations.reshape(-1, 1)), axis=1))
        graph_pretrain = Graph()
        el = defaultdict( #target_id
                        lambda: defaultdict( #source_id(
                        lambda: int # time
                    ))
        for i, j in tqdm(edge_index):
            el[i.item()][j.item()] = 1

        target_type = 'def'
        graph_pretrain.edge_list['def']['def']['def'] = el
        n = list(el.keys())
        # degree = np.zeros(np.max(n)+1)
        degree = np.zeros(n_feats.shape[0])
        for i in n:
            degree[i] = len(el[i])
        x = np.concatenate((n_feats, np.log(degree + 1e-5).reshape(-1, 1)), axis=-1)
        graph_pretrain.node_feature['def'] = pd.DataFrame({'emb': list(x)})

        # idx = np.arange(len(graph_pretrain.node_feature[target_type]))
        # np.random.shuffle(idx)
        idx = np.array(list(full_data.unique_nodes))

        graph_pretrain.pre_target_nodes = idx

        dill.dump(graph_pretrain, open('./dataset/graph_{}_{}_{}.pk'.format(args.mode, args.data, args.task_type), 'wb'))
    else:
        graph_pretrain: Graph = dill.load(open('./dataset/graph_pretrain_{}_{}.pk'.format(data_name, args.task_type), 'rb'))

        edge_index_train = torch.from_numpy(np.concatenate((train_data.sources.reshape(-1, 1), train_data.destinations.reshape(-1, 1)), axis=1))
        graph_down_train = Graph()
        el = defaultdict( #target_id
                        lambda: defaultdict( #source_id(
                        lambda: int # time
                    ))
        for i, j in tqdm(edge_index_train):
            el[i.item()][j.item()] = 1

        target_type = 'def'
        graph_down_train.edge_list['def']['def']['def'] = el
        n = list(el.keys())
        # degree = np.zeros(np.max(n)+1)
        degree = np.zeros(n_feats.shape[0])
        for i in n:
            degree[i] = len(el[i])
        x = np.concatenate((n_feats, np.log(degree + 1e-5).reshape(-1, 1)), axis=-1)
        graph_down_train.node_feature['def'] = pd.DataFrame({'emb': list(x)})

        # idx = np.arange(len(graph_down_train.node_feature[target_type]))
        # np.random.shuffle(idx)

        train_idx = np.array(list(train_data.unique_nodes))
        valid_idx = np.array(list(val_data.unique_nodes))
        test_idx = np.array(list(test_data.unique_nodes))

        graph_down_train.train_target_nodes = train_idx
        graph_down_train.valid_target_nodes = valid_idx
        graph_down_train.test_target_nodes = test_idx

        pretrain_idx_set = set(graph_pretrain.pre_target_nodes.tolist())
        train_idx_set = train_data.unique_nodes
        graph_pretrain.train_target_nodes = np.array(list(pretrain_idx_set & train_idx_set))

        dill.dump(graph_pretrain, open('./dataset/graph_pretrain_{}_{}.pk'.format(data_name, args.task_type), 'wb'))
        dill.dump(graph_down_train, open('./dataset/graph_{}_{}_{}.pk'.format(args.mode, args.data, args.task_type), 'wb'))

        dill.dump(train_data, open('./dataset/train_data_{}_{}_{}.pk'.format(args.mode, args.data, args.task_type), 'wb'))

        val_test_sampler = {
            'val_rand_sampler': val_rand_sampler, 
            'nn_val_rand_sampler': nn_val_rand_sampler, 
            'test_rand_sampler': test_rand_sampler, 
            'nn_test_rand_sampler': nn_test_rand_sampler
        }
        dill.dump(val_test_sampler, open('./dataset/val_test_sampler_{}_{}_{}.pk'.format(args.mode, args.data, args.task_type), 'wb'))

        val_test_data = {
            'val_data': val_data, 
            'new_node_val_data': new_node_val_data, 
            'test_data': test_data, 
            'new_node_test_data': new_node_test_data
        }
        dill.dump(val_test_data, open('./dataset/val_test_data_{}_{}_{}.pk'.format(args.mode, args.data, args.task_type), 'wb'))

        # full data

        # edge_index = torch.from_numpy(np.concatenate((full_data.sources.reshape(-1, 1), full_data.destinations.reshape(-1, 1)), axis=1))
        # graph_down = Graph()
        # el = defaultdict( #target_id
        #                 lambda: defaultdict( #source_id(
        #                 lambda: int # time
        #             ))
        # for i, j in tqdm(edge_index.t()):
        #     el[i.item()][j.item()] = 1

        # target_type = 'def'
        # graph_down.edge_list['def']['def']['def'] = el
        # n = list(el.keys())
        # degree = np.zeros(np.max(n)+1)
        # for i in n:
        #     degree[i] = len(el[i])
        # x = np.concatenate((n_feats, np.log(degree).reshape(-1, 1)), axis=-1)
        # graph_down.node_feature['def'] = pd.DataFrame({'emb': list(x)})

        # # idx = np.arange(len(graph_down.node_feature[target_type]))
        # # np.random.shuffle(idx)

        # # graph_down.pre_target_nodes = idx
        # dill.dump(graph_down, open('./dataset/full_graph_{}_{}_{}_{}.pk'.format(args.mode, args.data, args.task_type), 'wb'))


if __name__ == '__main__':
    main()