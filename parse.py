"""Command-line arguments for IDDIFF training and evaluation."""

import argparse


def parse_args():
    """Parse optimization, representation, graph, and diffusion settings."""
    parser = argparse.ArgumentParser(
        description="IDDIFF for Next POI Recommendation"
    )

    # Checkpoint and logging settings.
    parser.add_argument(
        '--path',
        type=str,
        default="../IDDIFF/code/checkpoints",
        help='path to save weights',
    )
    parser.add_argument('--log', type=str, default=None, help="log file path")
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--load', action='store_true', default=False)

    # Training and evaluation settings.
    parser.add_argument('--epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch', type=int, default=1024, help="training batch size")
    parser.add_argument('--testbatch', type=int, default=1024, help="evaluation batch size")
    parser.add_argument('--length', type=int, default=100, help="maximum trajectory length")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--decay', type=float, default=1e-3, help="L2 weight decay")
    parser.add_argument('--seed', type=int, default=9876, help='random seed')

    # Sequence-aware Trajectory Representation Module.
    parser.add_argument('--hidden', type=int, default=64, help="embedding dimension")
    parser.add_argument('--interval', type=int, default=256, help="number of spatio-temporal interval bins")
    parser.add_argument('--num_heads', type=int, default=2, help="number of attention heads")
    parser.add_argument('--window_size', type=int, default=3, help="sliding-window length for local subtrajectories")
    parser.add_argument('--guidance_w', type=float, default=0.5, help="intent prototype guidance weight")
    parser.add_argument('--dbscan_eps', type=float, default=0.5, help="DBSCAN neighborhood radius")
    parser.add_argument('--dbscan_min_samples', type=int, default=2, help="minimum DBSCAN cluster size")

    # POI Distance Graph Encoder.
    parser.add_argument('--layer', type=int, default=2, help="number of distance-aware GCN layers")

    # Intent Refinement Module and deterministic interpolation process.
    parser.add_argument('--zeta', type=float, default=0.2, help="intent refinement loss weight")
    parser.add_argument('--diffsize', type=int, default=1, help="diffusion horizon T")
    parser.add_argument('--stepsize', type=float, default=0.01, help="legacy diffusion step size")
    parser.add_argument('--beta_min', type=float, default=0.1, help="legacy minimum noise-schedule value")
    parser.add_argument('--beta_max', type=float, default=20, help="legacy maximum noise-schedule value")
    parser.add_argument('--interp_steps', type=int, default=10, help="number of deterministic reverse interpolation steps")

    # Recommendation scoring and inference.
    parser.add_argument('--alpha', type=float, default=2.0, help="weight of the sequential intent score")
    parser.add_argument('--sample_num', type=int, default=10, help="number of preference samples averaged at inference")
    parser.add_argument('--dropout', action='store_true', default=True, help="enable dropout")
    parser.add_argument('--dp', type=float, default=0.4, help="dropout probability")
    parser.add_argument('--patience', type=int, default=10, help="early-stopping patience")

    parser.add_argument(
        '--dataset',
        type=str,
        default='LA',
        help="available datasets: ['IST', 'JK', 'SP', 'NYC', 'LA']",
    )
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    return parser.parse_args()
