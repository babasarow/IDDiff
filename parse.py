import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="DiffDGMN for POI Recommendation")
    parser.add_argument('--path', type=str, default="../DiffDGMN/code/checkpoints", help='path to save weights')
    parser.add_argument('--log', type=str, default=None, help="log file path")
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)

    parser.add_argument('--epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch', type=int, default=1024, help="batch size for training")
    parser.add_argument('--testbatch', type=int, default=1024, help="batch size of users for testing")
    parser.add_argument('--length', type=int, default=100, help="max sequence length")
    parser.add_argument('--lr', type=float, default=0.001,  help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-3,  help="weight decay for L_2 normalizaton")
    parser.add_argument('--seed', type=int, default=9876,  help='random seed')
    
    parser.add_argument('--hidden', type=int, default=64, help="embedding size")
    parser.add_argument('--interval', type=int, default=256, help="temporal and locational intervals")
    parser.add_argument('--layer', type=int, default=2, help="layer num of GNN")
    parser.add_argument('--num_heads', type=int, default=2, help="number of heads")

    parser.add_argument('--zeta', type=float, default=0.2, help="Fisher divergence loss weight")
    parser.add_argument('--diffsize', type=int, default=1, help="diffusion size #T")
    parser.add_argument('--stepsize', type=float, default=0.01,  help="diffusion step size #dt")
    parser.add_argument('--beta_min', type=float, default=0.1, help="beta_min for noise schedules")
    parser.add_argument('--beta_max', type=float, default=20,  help="beta_max for noise schedules")

    # New hyperparameters
    parser.add_argument('--window_size', type=int, default=3, help="sliding window size")
    parser.add_argument('--guidance_w', type=float, default=0.5, help="guidance weight for clustering")
    parser.add_argument('--dbscan_eps', type=float, default=0.5, help="DBSCAN epsilon")
    parser.add_argument('--dbscan_min_samples', type=int, default=2, help="DBSCAN min samples")

    parser.add_argument('--dropout', action='store_true', default=True, help="using the dropout or not")
    parser.add_argument('--dp', type=float, default=0.4, help="dropout probability")
    parser.add_argument('--patience', type=int, default=10, help="early stop patience")

    # --- 新增: 添加权重系数 alpha，默认值为 2.0 (保持原逻辑) ---
    parser.add_argument('--alpha', type=float, default=2.0, help="weight for sequence matching score")
    
    # --- 新增: 推理时多重采样的次数，默认 10 ---
    parser.add_argument('--sample_num', type=int, default=10, help="number of samples for inference averaging")
    parser.add_argument('--interp_steps', type=int, default=10, help="number of discrete reverse interpolation steps")
    
    parser.add_argument('--dataset', type=str, default='LA', help="available datasets: ['IST', 'JK', 'SP', 'NYC', 'LA']") 
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    return parser.parse_args()
