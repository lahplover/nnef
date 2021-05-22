

###########################################
def get_phy_model_option():
    expid = 'exp94'
    # expid = 'exp93'
    embed_size = 32
    dim = 512
    seq_len = 11
    use_graph_net = ''
    n_layers = 2
    attn_heads = 4
    # use_graph_net = '--use_graph_net'

    model_option = f'--seq_len {seq_len} --mode CB --seq_type residue \
    --embed_size {embed_size} --dim {dim} --n_layers {n_layers} --attn_heads {attn_heads} \
    --use_phy_model {use_graph_net}'
    return expid, model_option


def get_prob_model_option():
    smooth_gaussian = '--smooth_gaussian --smooth_r 0.3 --smooth_angle 45 '
    loss_weights = '--coords_angle_loss_lamda 1 --profile_loss_lamda 10 --radius_loss_lamda 1 --start_id_loss_lamda 1 \
    --res_counts_loss_lamda 1 '

    # expid = 'exp212'
    # embed_size = 32
    # dim = 128
    # n_layers = 8
    # attn_heads = 4
    # seq_len = 14
    # use_model_os = '--use_model_os'
    # position_weights = ''
    # loss_weights = '--coords_angle_loss_lamda 1 --profile_loss_lamda 10 --radius_loss_lamda 1 --start_id_loss_lamda 1 \
    # --res_counts_loss_lamda 0.001 '

    # expid = 'exp211'
    # embed_size = 32
    # dim = 128
    # n_layers = 4
    # attn_heads = 4
    # seq_len = 14
    # use_model_os = ''
    # position_weights = ''

    # expid = 'exp208'
    # embed_size = 32
    # dim = 128
    # n_layers = 8
    # attn_heads = 4
    # seq_len = 14
    # use_model_os = '--use_model_os'
    # position_weights = ''

    # expid = 'exp209'
    # embed_size = 32
    # dim = 128
    # n_layers = 4
    # attn_heads = 4
    # seq_len = 14
    # use_model_os = ''
    # position_weights = '--use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 10'

    # expid = 'exp207'
    # embed_size = 32
    # dim = 128
    # n_layers = 4
    # attn_heads = 4
    # seq_len = 14
    # use_model_os = '--use_model_os'
    # position_weights = '--use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3'

    # expid = 'exp214'
    # embed_size = 32
    # dim = 128
    # n_layers = 4
    # attn_heads = 4
    # seq_len = 4
    # position_weights = ''
    # use_model_os = ''

    expid = 'exp215'
    embed_size = 32
    dim = 128
    n_layers = 4
    attn_heads = 4
    seq_len = 14
    position_weights = '--use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3'
    use_model_os = ''

    # expid = 'exp205'
    # embed_size = 32
    # dim = 128
    # n_layers = 4
    # attn_heads = 4
    # seq_len = 14
    # position_weights = '--use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3'
    # use_model_os = ''

    # expid = 'exp85'
    # # expid = 'exp79'  ## seq_len = 4
    # # expid = 'exp78'
    # # expid = 'exp72'
    # # expid = 'exp73'
    # # expid = 'exp70'
    # #
    # embed_size = 32
    # dim = 128
    # n_layers = 2
    # attn_heads = 4
    # seq_len = 11
    #
    # expid = 'exp69'
    # embed_size = 64
    # dim = 256
    # n_layers = 4
    # attn_heads = 8
    # seq_len = 14
    #
    # expid = 'exp66'
    # embed_size = 32
    # dim = 128
    # n_layers = 2
    # attn_heads = 4
    # seq_len = 14
    #
    # expid = 'exp65'
    # embed_size = 64
    # dim = 256
    # n_layers = 4
    # attn_heads = 8
    # seq_len = 14

    model_option = f'{use_model_os} --seq_len {seq_len} --mode CB --seq_type residue \
    --embed_size {embed_size} --dim {dim} --n_layers {n_layers} --attn_heads {attn_heads} \
    --mixture_r 2 --mixture_angle 3 --mixture_seq 1 {smooth_gaussian} {loss_weights} {position_weights}'
    return expid, model_option
###########################################


expid, model_option = get_prob_model_option()
# expid, model_option = get_phy_model_option()


mf = open(f'test_{expid}.sh', 'wt')

# decoys
for decoy_set in ['3DRobot_set', 'casp13', 'casp14', 'casp11']:
    mf.write(f"""
# score decoys {decoy_set}
python erf/decoy_score.py {model_option} \
--load_exp runs/{expid} --decoy_set {decoy_set} --decoy_loss_dir decoy_loss_{expid}\n""")

for decoy_set in ['4state_reduced', 'lattice_ssfit', 'lmds', 'lmds_v2']:
    mf.write(f"""
# score decoys RU {decoy_set}
python erf/decoy_ru_score.py {model_option} \
--load_exp runs/{expid} --decoy_set {decoy_set} --decoy_loss_dir {expid}\n""")

for decoy_set in ['random', 'shuffle', 'type2', 'type2LD', 'type9', 'polyAA']:
    mf.write(f"""
# score sequence decoys: {decoy_set}
python erf/decoy_seq.py {model_option} \
--load_exp runs/{expid} --decoy_set {decoy_set} --decoy_loss_dir {expid}\n""")

mf.write(f"""
# score MD decoys
python erf/decoy_md_score.py {model_option} \
--load_exp runs/{expid} --decoy_set decoy --decoy_loss_dir {expid}\n""")

# run Brownian dynamics
mf.write(f"""
# run Brownian dynamics
python erf/fold_os.py {model_option} \
--load_exp runs/{expid} --fold_engine dynamics --x_type cart --L 500 --T_max 0.03 --lr 2e-3 \
--save_dir {expid}dynamics_val_deep\n""")

# make decoys by moving the loops
mf.write(f"""
# make decoys by moving the loops
python erf/make_decoy.py {model_option} \
--load_exp runs/{expid} --fold_engine anneal --L 1000 --T_max 2.25 --T_min 0.9 \
--save_dir {expid}anneal_val_deep_loop\n""")

# design: mutations & sequence redesign
mf.write(f"""
# design: mutations
python erf/design_sample.py {model_option} \
--load_exp runs/{expid} --fold_engine mutation --save_dir {expid}mutation_val_deep\n""")

mf.write(f"""
# design: sequence redesign
python erf/design_sample_deep.py {model_option} \
--load_exp runs/{expid} --fold_engine anneal --random_init --L 1000 --T_max 1.5 --T_min 0.6 \
--save_dir {expid}anneal_val_deep\n""")

mf.write(f"""
# score decoys of the stability dataset
python erf/score_stability.py {model_option} \
--load_exp runs/{expid} --decoy_loss_dir {expid}\n""")


# fold anneal

# fold grad

mf.close()


def brownian_dynamics_grid_search():
    smooth_gaussian = '--smooth_gaussian --smooth_r 0.3 --smooth_angle 45 '
    loss_weights = '--coords_angle_loss_lamda 1 --profile_loss_lamda 10 --radius_loss_lamda 1 --start_id_loss_lamda 1 \
    --res_counts_loss_lamda 1 '

    expid = 'exp205'
    embed_size = 32
    dim = 128
    n_layers = 4
    attn_heads = 4
    seq_len = 14
    position_weights = '--use_position_weights --cen_seg_loss_lamda 1 --oth_seg_loss_lamda 3'
    use_model_os = ''

    model_option = f'{use_model_os} --seq_len {seq_len} --mode CB --seq_type residue \
    --embed_size {embed_size} --dim {dim} --n_layers {n_layers} --attn_heads {attn_heads} \
    --mixture_r 2 --mixture_angle 3 --mixture_seq 1 {smooth_gaussian} {loss_weights} {position_weights}'

    with open(f'test_{expid}_bd_grid.sh', 'wt') as mf:
        flag = 10
        for T_max in [0.03, 0.06, 0.1, 0.15, 0.2]:
            for lr in [5e-3, 1e-2, 2e-2, 3.5e-2, 5e-2]:
                flag += 1
                mf.write(f"""
# run Brownian dynamics
python erf/fold_os.py {model_option} \
--load_exp runs/{expid} --fold_engine dynamics --x_type cart --L 10000 --T_max {T_max} --lr {lr} \
--save_dir {expid}dynamics_val_deep{flag}\n\n""")

    with open(f'scp_bd_grid.sh', 'wt') as mf:
        flag = 10
        for T_max in [0.03, 0.06, 0.1, 0.15, 0.2]:
            for lr in [5e-3, 1e-2, 2e-2, 3.5e-2, 5e-2]:
                flag += 1
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/3KXT_A_rmsf.pdf '
                         f'3KXT_A_rmsf_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/3KXT_A_rmsd.pdf '
                         f'3KXT_A_rmsd_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/3KXT_score.pdf '
                         f'3KXT_score_{flag}_Tmax{T_max}_lr{lr}.pdf\n')

    with open(f'test_{expid}_bd_grid2.sh', 'wt') as mf:
        flag = 100
        for T_max in [0.05, 0.06, 0.07, 0.08, .09, 0.1, 0.11, 0.12]:
            for lr in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
                flag += 1
                mf.write(f"""
# run Brownian dynamics
python erf/fold_os.py {model_option} \
--load_exp runs/{expid} --fold_engine dynamics --x_type cart --L 10000 --T_max {T_max} --lr {lr} \
--save_dir {expid}dynamics_val_deep{flag}\n\n""")

    with open(f'scp_bd_grid2.sh', 'wt') as mf:
        flag = 100
        for T_max in [0.05, 0.06, 0.07, 0.08, .09, 0.1, 0.11, 0.12]:
            for lr in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
                flag += 1
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/3KXT_A_rmsf.pdf '
                         f'3KXT_A_rmsf_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/3KXT_A_rmsd.pdf '
                         f'3KXT_A_rmsd_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/3KXT_score.pdf '
                         f'3KXT_score_{flag}_Tmax{T_max}_lr{lr}.pdf\n')

    with open(f'test_{expid}_bd_grid3.sh', 'wt') as mf:
        flag = 200
        T_max_list = [0.06, 0.06, 0.06, 0.08, 0.08, 0.08]
        lr_list = [0.01, 0.015, 0.02, 0.015, 0.02, 0.03]
        for T_max, lr in zip(T_max_list, lr_list):
            flag += 1
            mf.write(f"""
# run Brownian dynamics
python erf/fold_os.py {model_option} \
--load_exp runs/{expid} --fold_engine dynamics --x_type cart --L 10000 --T_max {T_max} --lr {lr} \
--save_dir {expid}dynamics_val_deep{flag}\n\n""")

    with open(f'scp_bd_grid3.sh', 'wt') as mf:
        flag = 200
        T_max_list = [0.06, 0.06, 0.06, 0.08, 0.08, 0.08]
        lr_list = [0.01, 0.015, 0.02, 0.015, 0.02, 0.03]
        for T_max, lr in zip(T_max_list, lr_list):
            flag += 1
            pdbs = "1ZZK,2MCM,2VIM,3FBL,3IPZ,3KXT,3NGP,3P0C,3SNY,3SOL,3VI6,4M1X,4O7Q,4QRL,5CYB,5JOE,6H8O"
            pdb_list = pdbs.split(',')
            for pdb in pdb_list:
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsf.pdf '
                         f'{pdb}_A_rmsf_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsf_simple.pdf '
                         f'{pdb}_A_rmsf_simple_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsd.pdf '
                         f'{pdb}_A_rmsd_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_score.pdf '
                         f'{pdb}_score_{flag}_Tmax{T_max}_lr{lr}.pdf\n')

    flag = 300
    lr_list = [1e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 4e-2]
    for i, lr in enumerate(lr_list):
        with open(f'test_{expid}_bd_grid4_{i}.sh', 'wt') as mf:
            for T_max in [0.01, 0.03, 0.06, 0.1, 0.15]:
                flag += 1
                mf.write(f"""
# run Brownian dynamics
python erf/fold_os.py {model_option} \
--load_exp runs/{expid} --fold_engine dynamics --x_type cart --L 20000 --T_max {T_max} --lr {lr} \
--save_dir {expid}dynamics_val_deep{flag}\n\n""")

    import numpy as np
    with open(f'scp_bd_grid4.sh', 'wt') as mf:
        T_max_list = [0.01, 0.03, 0.06, 0.1, 0.15]
        lr_list = [1e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 4e-2]
        flag_dict = {}
        flag = 300
        for lr in lr_list:
            for T_max in T_max_list:
                flag += 1
                flag_dict[flag] = (lr, T_max)

        flag_list = np.append(np.array([301, 302, 306, 307]), np.arange(10) + 311)
        for flag in flag_list:
            lr, T_max = flag_dict[flag]
            pdbs = "1ZZK,2MCM,2VIM,3FBL,3IPZ,3KXT,3NGP,3P0C,3SNY,3SOL,3VI6,4M1X,4O7Q,4QRL,5CYB,5JOE,6H8O"
            pdb_list = pdbs.split(',')
            for pdb in pdb_list:
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsf.pdf '
                         f'{pdb}_A_rmsf_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsf_simple.pdf '
                         f'{pdb}_A_rmsf_simple_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsd.pdf '
                         f'{pdb}_A_rmsd_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_score.pdf '
                         f'{pdb}_score_{flag}_Tmax{T_max}_lr{lr}.pdf\n')

    flag_dict = {4: (0.05, 0.02), 5: (0.05, 0.03), 7: (0.035, 0.1), 8: (0.006, 0.05)}
    ## second grid search
    flag = 200
    T_max_list = [0.06, 0.06, 0.06, 0.08, 0.08, 0.08]
    lr_list = [0.01, 0.015, 0.02, 0.015, 0.02, 0.03]
    for T_max, lr in zip(T_max_list, lr_list):
        flag += 1
        flag_dict[flag] = (lr, T_max)
    ## third grid search
    T_max_list = [0.01, 0.03, 0.06, 0.1, 0.15]
    lr_list = [1e-3, 3e-3, 5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 4e-2]
    flag = 300
    for lr in lr_list:
        for T_max in T_max_list:
            flag += 1
            flag_dict[flag] = (lr, T_max)
    with open(f'scp_bd_grid_3_4.sh', 'wt') as mf:
        for flag in flag_dict.keys():
            lr, T_max = flag_dict[flag]
            pdbs = "1ZZK,2MCM,2VIM,3FBL,3IPZ,3KXT,3NGP,3P0C,3SNY,3SOL,3VI6,4M1X,4O7Q,4QRL,5CYB,5JOE,6H8O"
            pdb_list = pdbs.split(',')
            for pdb in pdb_list:
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsf.pdf '
                         f'{pdb}_A_rmsf_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsf_simple.pdf '
                         f'{pdb}_A_rmsf_simple_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_A_rmsd.pdf '
                         f'{pdb}_A_rmsd_{flag}_Tmax{T_max}_lr{lr}.pdf\n')
                mf.write(f'scp $cbio:~/bio/erf/data/fold/exp205dynamics_val_deep{flag}/{pdb}_score.pdf '
                         f'{pdb}_score_{flag}_Tmax{T_max}_lr{lr}.pdf\n')










