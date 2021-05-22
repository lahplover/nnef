import argparse


def get_common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_exp", type=str, default=None)

    parser.add_argument("--mode", type=str, default='CB', help='CA / CB / CAS')

    parser.add_argument("--seq_len", type=int, default=14)
    parser.add_argument("--seq_type", type=str, default='profile')
    parser.add_argument("--residue_type_num", type=int, default=20,
                        help='number of residue types used in the sequence vocabulary')
    parser.add_argument("--seq_factor", type=float, default=0.5)
    parser.add_argument("--noise_factor", type=float, default=0.001)

    parser.add_argument("--dist_mask", action='store_true', default=False)
    parser.add_argument("--dist_cutoff", type=float, default=10)

    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--attn_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--n_layers", type=int, default=2)

    parser.add_argument("--use_model_os", action='store_true', default=False)
    parser.add_argument("--use_phy_model", action='store_true', default=False)
    parser.add_argument("--use_graph_net", action='store_true', default=False)

    parser.add_argument("--mixture_r", type=int, default=2, help="num of Gaussian mixtures for radius")
    parser.add_argument("--mixture_angle", type=int, default=10, help="num of Gaussian mixtures for angles")
    parser.add_argument("--mixture_seq", type=int, default=3, help="num of Gaussian mixtures for sequences")
    parser.add_argument("--mixture_res_counts", type=int, default=1, help="num of Gaussian mixtures for res counts")

    parser.add_argument("--smooth_gaussian", action='store_true', default=False,
                        help='smooth the gaussian mixture function of r, theta, phi.')
    parser.add_argument("--smooth_r", type=float, default=0.3, help='minimum r_sigma = 0.3A')
    parser.add_argument("--smooth_angle", type=float, default=15.0, help='minimum angle_sigma=15 deg')
    parser.add_argument("--random_ref", action='store_true', default=False,
                        help='use random distribution as reference prob. E=-log(P/Pref)')

    parser.add_argument("--reduction", type=str, default='sum_all',
                        help='if reduction == keep_batch_dim, return loss of each batch member')

    parser.add_argument("--coords_angle_loss_lamda", type=float, default=1, help="weight of coords_angle_loss")
    parser.add_argument("--profile_loss_lamda", type=float, default=1, help="weight of profile loss")
    parser.add_argument("--radius_loss_lamda", type=float, default=1, help="weight of radius loss")
    parser.add_argument("--start_id_loss_lamda", type=float, default=1, help="weight of start_id loss")
    parser.add_argument("--res_counts_loss_lamda", type=float, default=1, help="weight of res_counts loss")

    parser.add_argument("--use_position_weights", action='store_true', default=False, help="use position weights")
    parser.add_argument("--cen_seg_loss_lamda", type=float, default=1, help="weight of the central segment loss")
    parser.add_argument("--oth_seg_loss_lamda", type=float, default=1, help="weight of the other segments loss")

    parser.add_argument("--profile_prob", action='store_true', default=False, help="use target profile as weight")

    parser.add_argument("--device", type=str, default='cuda')

    parser.add_argument("--debug", action='store_true', default=False)

    return parser


def get_local_gen_parser():
    parser = get_common_parser()
    parser.add_argument("--data_flag", type=str, default='train_small')
    parser.add_argument("--val_data_flag", type=str, default=None)
    parser.add_argument("--test_data_flag", type=str, default=None)
    parser.add_argument("--chimeric", action='store_true', default=False)
    parser.add_argument("--chimeric_pdb", type=str, default='train_small_pdb')
    parser.add_argument("--struct_seq_id", type=str, default='struct_seq_id_CB')
    parser.add_argument("--total_num_samples", type=int, default=1000000, help='used in WeightedRandomSampler')
    parser.add_argument("--no_homology", action='store_true', default=False)

    parser.add_argument("--save_path", type=str, default='./runs/')

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", default=(0.9, 0.99))
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--n_warmup_steps", type=int, default=5000)
    parser.add_argument("--steps_decay_scale", type=int, default=10000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)

    return parser


def get_fold_parser():
    parser = get_common_parser()
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--fold_engine", type=str, default='anneal', help='anneal / grad / dynamics')

    parser.add_argument("--T_max", type=float, default=0.1)
    parser.add_argument("--T_min", type=float, default=0.01)
    parser.add_argument("--L", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate used in grad and dynamics')

    parser.add_argument("--anneal_type", type=str, default='int_one', help='int_one / frag')
    parser.add_argument("--ic_move_std", type=float, default=2, help='standard deviation of the internal angle move')
    parser.add_argument("--x_type", type=str, default='cart', help='cart / internal / mixed, for grad minimizer')
    parser.add_argument("--random_init", action='store_true', default=False)
    parser.add_argument("--use_rg", action='store_true', default=False)

    parser.add_argument("--seq_move_type", type=str, default='mutate_one', help='mutate_one / swap_one')

    parser.add_argument("--relax", action='store_true', default=False, help='relax the decoy in the new energy function')
    parser.add_argument("--relax_steps", type=int, default=200)

    return parser


def get_decoy_parser():
    parser = get_local_gen_parser()

    parser.add_argument("--decoy_set", type=str, default='casp11')
    parser.add_argument("--decoy_loss_dir", type=str, default='decoy_loss')

    parser.add_argument("--static_decoy", action='store_true', default=False, help='use precomputed static decoys')

    parser.add_argument("--relax", action='store_true', default=False, help='relax the decoy in the new energy function')
    parser.add_argument("--relax_steps", type=int, default=200)

    return parser


def get_train_fold_parser():
    parser = get_common_parser()
    parser.add_argument("--run_folding", action='store_true', default=False)

    parser.add_argument("--t_noise", type=float, default=1e-3)
    parser.add_argument("--lr_dynamics", type=float, default=1e-3, help='learning rate used in dynamics')
    parser.add_argument("--L", type=int, default=1000)
    parser.add_argument("--lr_ml", type=float, default=1e-3, help='learning rate used in machine learning')
    parser.add_argument("--weight_decay", type=float, default=1e-5, help='L2 weight regularization')

    parser.add_argument("--x_type", type=str, default='cart', help='cart / internal / mixed, for grad minimizer')
    parser.add_argument("--random_init", action='store_true', default=False)
    parser.add_argument("--unfold_init", action='store_true', default=False)
    parser.add_argument("--fast_dzdx", action='store_true', default=False)
    parser.add_argument("--use_loss_grad", action='store_true', default=False)

    parser.add_argument("--data_flag", type=str, default='train_small')
    parser.add_argument("--val_data_flag", type=str, default=None)
    parser.add_argument("--test_data_flag", type=str, default=None)

    parser.add_argument("--total_num_samples", type=int, default=1000000, help='used in WeightedRandomSampler')

    parser.add_argument("--n_warmup_steps", type=int, default=5000)
    parser.add_argument("--steps_decay_scale", type=int, default=10000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--save_path", type=str, default='./runs/', help='path to save learned models')

    return parser


def parse_args_and_arch(parser):
    args = parser.parse_args()
    return args
