import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from scipy import stats
import h5py
import os


######################################################
amino_acids = pd.read_csv('data/amino_acids.csv')
idx2aa = {x-1: y for x, y in zip(amino_acids.idx, amino_acids.AA)}
aa = amino_acids['AA'].values

aa_freq = pd.read_csv('data/aa_freq.csv')
freq_dict = {x-1: y for x, y in zip(aa_freq.idx, aa_freq.freq)}

ordered_aa = 'AVILMFWYGPSTCQNRKHED'
new_aa_idx = {y: x for x, y in enumerate(ordered_aa)}
map_aa = {x-1: new_aa_idx[y] for x, y in zip(amino_acids.idx, amino_acids.AA)}

# for i in range(20):
#     print(i, map_aa[i], aa[i], ordered_aa[map_aa[i]])


mode = 'CB'
# pdb_selected = ['1BPI_1_A', '1FME_1_A', '2A3D_1_A', '2HBA_1_A', '2JOF_1_A', '2P6J_1_A', '2WXC_1_A']
# protein_sample = pd.read_csv('../fold/protein_sample/sample.csv')
# pdb_selected = protein_sample['pdb_id'].values
protein_sample = pd.read_csv('data/design/cullpdb_val_deep/sample.csv')
# protein_sample = pd.read_csv('data/design/cullpdb_val_deep/sample_train.csv')
pdb_selected = protein_sample['pdb'].values

# exp_flag = 'exp54_ps_'
# exp_flag = 'exp61'
# exp_flag = 'exp78'
exp_flag = 'exp205'
# exp_flag = 'exp213'

# pdb_selected_used = []
# for pdb_id in pdb_selected:
#     if os.path.exists(f'{exp_flag}anneal_ps/{pdb_id}_profile.h5'):
#         pdb_selected_used.append(pdb_id)
# pdb_selected = pdb_selected_used


######################################################
def single_mutation():
    # single site mutations scan; calculate mutation patterns and recovery rate
    root_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep/{exp_flag}mutation_val_deep'

    mut_matrix = np.zeros((20, 20))
    recovery_mutant = []

    for pdb_id in pdb_selected:
        data = h5py.File(f'{root_dir}/{pdb_id}_profile.h5', 'r')
        # residue_energy = data['wt_residue_energy'][()]
        mutant_energy = data['mutant_energy'][()]
        seq = data['seq'][()].astype(np.int)
        # profile = data['profile'][()]
        seq_mutant = np.argmin(mutant_energy, axis=1)

        recovery_fraction = np.sum(seq_mutant == seq) / float(len(seq))
        recovery_mutant.append(recovery_fraction)

        for i in range(seq.shape[0]):
            # mut_matrix[seq[i], seq_mutant[i]] += 1
            mut_matrix[map_aa[seq[i]], map_aa[seq_mutant[i]]] += 1
        # print(seq, seq_mutant)

    # plot the mutation results
    fig = pl.figure()
    mut_matrix_freq = mut_matrix / mut_matrix.sum(axis=1)[:, None]
    # pl.imshow(mut_matrix_freq, cmap='Greys')
    pl.imshow(mut_matrix_freq, cmap='jet')
    pl.xlabel('mutated residue')
    pl.ylabel('native residue')
    pl.xticks(np.arange(20), labels=ordered_aa)
    pl.yticks(np.arange(20), labels=ordered_aa)
    pl.colorbar()
    pl.title('single residue mutation')
    pl.savefig(f'{root_dir}/single_residue_mutation.pdf')


######################################################
def seq_design():
    # full sequence redesign; calculate mutation patterns and recovery rate
    # root_dir = f'/home/hyang/bio/erf/data/design/{exp_flag}anneal_ps'
    root_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep/{exp_flag}swap_val_deep'

    mut_matrix_anneal = np.zeros((20, 20))

    seq_len = []
    recovery = []
    recovery_res = np.zeros(20)
    count_res = np.zeros(20)
    seq_best_all = np.array([])

    # if calculate group mutation patterns.
    # num_res_type = 9
    # aa_types = pd.read_csv('../aa_types.csv')
    # res_type = aa_types[f'type{num_res_type}']
    # aa_types_vocab = {x - 1: y for x, y in zip(aa_types.idx, res_type)}
    #
    # recovery_gp = []
    # recovery_res_gp = np.zeros(num_res_type)
    # count_res_gp = np.zeros(num_res_type)

    for pdb_id in pdb_selected:
        # pdb_id_bead = pdb_id.split('_')[0] + '_' + pdb_id.split('_')[2]
        # df_beads = pd.read_csv(f'protein_sample/{pdb_id_bead}_bead.csv')
        # df_beads = pd.read_csv(f'protein_sample/{pdb_id}_bead.csv')
        #
        # if mode == 'CA':
        #     coords = df_beads[['xca', 'yca', 'zca']].values
        # elif mode == 'CB':
        #     coords = df_beads[['xcb', 'ycb', 'zcb']].values
        # elif mode == 'CAS':
        #     coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
        # else:
        #     raise ValueError('mode should be CA / CB / CAS.')

        data_anneal = h5py.File(f'{root_dir}/{pdb_id}_profile.h5', 'r')
        designed_seq = data_anneal['profile'][()]
        seq_best = designed_seq[1]
        seq_best_all = np.append(seq_best_all, seq_best)
        seq_native = designed_seq[0]
        recovery_fraction = np.sum(seq_best == seq_native) / float(len(seq_native))
        recovery.append(recovery_fraction)
        seq_len.append(len(seq_native))

        # mutation patterns
        for i in range(seq_native.shape[0]):
            # mut_matrix_anneal[seq_native[i], seq_best[i]] += 1
            mut_matrix_anneal[map_aa[seq_native[i]], map_aa[seq_best[i]]] += 1

        # recovery per residue type
        for i in range(20):
            idx = (seq_native == i)
            recovery_res[i] += np.sum(seq_best[idx] == seq_native[idx])
            count_res[i] += len(seq_native[idx])

        # # grouped seq
        # seq_native_gp = np.array([aa_types_vocab[x] for x in seq_native])
        # seq_best_gp = np.array([aa_types_vocab[x] for x in seq_best])
        #
        # recovery_fraction_gp = np.sum(seq_best_gp == seq_native_gp) / float(len(seq_native_gp))
        # recovery_gp.append(recovery_fraction_gp)
        #
        # for i in range(num_res_type):
        #     idx = (seq_native_gp == i)
        #     recovery_res_gp[i] += np.sum(seq_best_gp[idx] == seq_native_gp[idx])
        #     count_res_gp[i] += len(seq_native_gp[idx])

        # print(pdb_id, len(seq_native), recovery_fraction, recovery_fraction_gp)
        if len(seq_native) < 100:
            print(pdb_id, len(seq_native), recovery_fraction)
            print(seq_native, seq_best)

        # write_pdb(seq, coords, pdb_id, 'native')
        # write_pdb(seq_best, coords, pdb_id, 'best')
        # write_pdb(seq_mutant, coords, pdb_id, 'mutant')

    df = pd.DataFrame({'pdb': pdb_selected, 'seq_len': seq_len,
                       'recovery': recovery,
                       # 'recovery_gp': recovery_gp,
                       # 'recovery_mutant': recovery_mutant
                       })
    df.to_csv(f'{root_dir}/recovery.csv', index=False)

    # plot the full sequence design results
    # check the recovery fraction per residue or residue group
    fig = pl.figure()
    pl.plot(np.arange(20), recovery_res / count_res)
    pl.xticks(np.arange(20), labels=aa)
    pl.title('residue recovery fraction in seq swap')
    pl.savefig(f'{root_dir}/full_seq_design_residue_recovery.pdf')

    # fig = pl.figure()
    # pl.plot(np.arange(num_res_type), recovery_res_gp / count_res_gp)
    #
    # res_type_labels_dict = {9: ['AVILM', 'ST', 'C', 'FWY', 'P', 'NQ', 'G', 'HKR', 'DE'],
    #                         2: ['AVILMFGPSTC', 'DEHKRNQWY'],
    #                         3: ['AVILMGP', 'CNQSTFWY', 'DEHKR'],
    #                         5: ['AVILMP', 'CSTNQ', 'DEHKR', 'FWY', 'G'],
    #                         7: ['AVILM', 'CST', 'DEHKR', 'FWY', 'P', 'NQ', 'G']}
    # res_type_labels = res_type_labels_dict[num_res_type]
    # pl.xticks(np.arange(num_res_type), labels=res_type_labels)

    fig = pl.figure()
    mut_matrix_anneal_freq = mut_matrix_anneal / mut_matrix_anneal.sum(axis=1)[:, None]
    pl.imshow(mut_matrix_anneal_freq, cmap='jet')
    pl.xlabel('mutated residue')
    pl.ylabel('native residue')
    pl.xticks(np.arange(20), labels=ordered_aa)
    pl.yticks(np.arange(20), labels=ordered_aa)
    pl.colorbar()
    pl.title('full seq redesign')
    pl.savefig(f'{root_dir}/full_seq_design_residue_use.pdf')

    # pl.figure()
    # pl.plot(df['seq_len'], df['recovery'], 'bo')

    fig = pl.figure()
    res_all = pd.value_counts(seq_best_all)
    res_aa_freq = res_all / np.sum(res_all)
    for i, count in zip(res_aa_freq.index, res_aa_freq):
        pl.scatter(i, count)
    pl.xticks(np.arange(20), labels=aa)
    pl.title('residue use frequency')
    pl.savefig(f'{root_dir}/full_seq_design_residue_use_frequency.pdf')


######################################################
def deep_seq_design():
    # deep full sequence redesign; calculate mutation patterns and recovery rate
    # root_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep/{exp_flag}anneal_val_deep'
    root_dir = f'data/design/cullpdb_val_deep/{exp_flag}anneal_val_deep'

    mut_matrix_anneal = np.zeros((20, 20))
    native_aa_all = []
    design_aa_all = []

    seq_len = []
    recovery = []
    pdb_id_all = []
    recovery_res = np.zeros(20)
    count_res = np.zeros(20)

    num = 100
    for pdb_id in pdb_selected:
        seq_best_all = []
        for j in range(num):
            data_anneal = h5py.File(f'{root_dir}/{pdb_id}_profile_{j}.h5', 'r')
            designed_seq = data_anneal['profile'][()]
            seq_best = designed_seq[1]
            seq_best_all.append(seq_best)

            seq_native = designed_seq[0]
            recovery_fraction = np.sum(seq_best == seq_native) / float(len(seq_native))
            recovery.append(recovery_fraction)
            seq_len.append(len(seq_native))
            pdb_id_all.append(pdb_id)

            # mutation patterns
            for i in range(seq_native.shape[0]):
                # mut_matrix_anneal[seq_native[i], seq_best[i]] += 1
                mut_matrix_anneal[map_aa[seq_native[i]], map_aa[seq_best[i]]] += 1
                native_aa_all.append(map_aa[seq_native[i]])
                design_aa_all.append(map_aa[seq_best[i]])

            # recovery per residue type
            for i in range(20):
                idx = (seq_native == i)
                recovery_res[i] += np.sum(seq_best[idx] == seq_native[idx])
                count_res[i] += len(seq_native[idx])

        # write fasta file of the best designed sequences
        with open(f'{root_dir}/{pdb_id}_seq_best.fasta', 'w') as mf:
            s = ''.join([idx2aa[x] for x in seq_native])
            mf.write(f'>0\n{s}\n')
            for j in range(len(seq_best_all)):
                s = ''.join([idx2aa[x] for x in seq_best_all[j]])
                mf.write(f'>{j+1}\n')
                mf.write(f'{s}\n')

    df = pd.DataFrame({'pdb': pdb_id_all, 'seq_len': seq_len,
                       'recovery': recovery,
                       })
    df.to_csv(f'{root_dir}/recovery.csv', index=False)

    # save the mutation matrix
    np.save(f'data/design/cullpdb_val_deep/{exp_flag}anneal_val_deep/mut_matrix_anneal.npy', mut_matrix_anneal)
    df = pd.DataFrame({'native_aa': native_aa_all, 'design_aa': design_aa_all})
    df.to_csv(f'{root_dir}/native_design_aa.csv', index=False)


    fig = pl.figure()
    pl.plot(df['seq_len'], df['recovery'], 'bo')
    pl.title('full seq redesign')
    pl.savefig(f'{root_dir}/full_seq_design_seqlen_recovery.pdf')

    # df = pd.read_csv(f'{root_dir}/recovery.csv')
    fig = pl.figure()
    pl.hist(df['recovery'], bins=np.arange(10)*0.05 + 0.05)
    # pl.title('full seq redesign')
    pl.xlabel('native sequence recovery fraction')
    pl.ylabel('N')
    pl.savefig(f'{root_dir}/full_seq_design_recovery_hist.pdf')

    fig = pl.figure()
    pl.plot(np.arange(20), recovery_res / count_res)
    pl.xticks(np.arange(20), labels=aa)
    pl.title('residue recovery fraction in full seq redesign')
    pl.savefig(f'{root_dir}/full_seq_design_residue_recovery.pdf')

    fig = pl.figure()
    mut_matrix_anneal_freq = mut_matrix_anneal / mut_matrix_anneal.sum(axis=1)[:, None]
    pl.imshow(mut_matrix_anneal_freq, cmap='jet')
    pl.xlabel('mutated residue')
    pl.ylabel('native residue')
    pl.xticks(np.arange(20), labels=ordered_aa)
    pl.yticks(np.arange(20), labels=ordered_aa)
    pl.colorbar()
    pl.title('full seq redesign')
    pl.savefig(f'{root_dir}/full_seq_design_residue_use.pdf')

    fig = pl.figure()
    res_all = np.concatenate(seq_best_all).flatten()
    # res_all = pd.value_counts(np.concatenate(seq_best_all).flatten())
    # res_aa_freq = res_all / np.sum(res_all)
    # for i, count in zip(res_aa_freq.index, res_aa_freq):
    #     pl.scatter(i, count)
    aa_bins = np.arange(21) - 0.5
    pl.hist(res_all, bins=aa_bins, histtype='step')
    pl.xticks(np.arange(20), labels=aa)
    pl.title('residue use frequency')
    pl.savefig(f'{root_dir}/full_seq_design_residue_use_frequency.pdf')


def get_fasta():
    pdb_id = '1ZZK'
    pdb_id = '5CYB'
    df = pd.read_csv(f'{pdb_id}_A.chimeric')
    seq = df['seq'].values
    with open(f'{pdb_id}_A_chimeric.fasta', 'w') as f:
        for i, s in enumerate(seq):
            f.write(f'>{i}\n')
            f.write(f'{s[63:]}\n')


def get_pdb():
    root_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep'
    mode = 'CB'
    # amino_acids = pd.read_csv('amino_acids.csv')
    # vocab = {x - 1: y for x, y in zip(amino_acids.idx, amino_acids.AA3C)}

    for pdb_id in pdb_selected:
        df_beads = pd.read_csv(f'{root_dir}/{pdb_id}_bead.csv')
        seq = df_beads['group_name']

        if mode == 'CA':
            coords = df_beads[['xca', 'yca', 'zca']].values
        elif mode == 'CB':
            coords = df_beads[['xcb', 'ycb', 'zcb']].values
        elif mode == 'CAS':
            coords = (df_beads[['xca', 'yca', 'zca']].values + df_beads[['xs', 'ys', 'zs']].values) / 2
        else:
            raise ValueError('mode should be CA / CB / CAS.')

        num = np.arange(coords.shape[0])
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        with open(f'{root_dir}/{pdb_id}_native.pdb', 'wt') as mf:
            for i in range(len(num)):
                mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')


def get_sasa():
    import mdtraj as md
    save_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep/'
    protein_sample = pd.read_csv('/home/hyang/bio/erf/data/design/cullpdb_val_deep/sample.csv')
    pdb_selected = protein_sample['pdb'].values
    for p in pdb_selected:
        pdb_id = p[:4]
        structure = md.load(f'/home/hyang/bio/openmm/data/{pdb_id}/production_T300.pdb')
        topology = structure.topology
        trj = md.load(f'/home/hyang/bio/openmm/data/{pdb_id}/production_T300.dcd', top=topology)
        topology.create_standard_bonds()
        trj = trj.remove_solvent()
        sasa = md.shrake_rupley(trj, mode='residue')
        sasa *= 100
        fig = pl.figure()
        for i in range(sasa.shape[0]):
            pl.plot(sasa[i])
        pl.xlabel('Residue Number')
        pl.ylabel('SASA (A^2)')
        pl.savefig(f'{save_dir}/{p}_sasa.pdf')
        pl.close(fig)

        df = pd.read_csv(f'{save_dir}/{p}_bead.csv')
        df['sasa'] = np.mean(sasa, axis=0)
        df.to_csv(f'{save_dir}/{p}_bead_sasa.csv')


def get_conservation(seq_best_all):
    # seq_best_all (L, N), conserv_score (L,), L is seq len
    L, N = seq_best_all.shape
    conserv_score = np.zeros(L)
    for i in range(L):
        aa_pseudo_count = np.arange(20)
        aa_list = np.append(seq_best_all[i], aa_pseudo_count)
        aa_prob = pd.value_counts(aa_list, normalize=True).values
        conserv_score[i] = -1.0 * np.sum(aa_prob * np.log(aa_prob))  # Shannon entropy
    print(conserv_score)
    return conserv_score


def plot_sasa_designed_seq():
    amino_acids = pd.read_csv('/home/hyang/bio/erf/data/amino_acids_msasa.csv')
    idx2aa = {x - 1: y for x, y in zip(amino_acids.idx, amino_acids.AA)}
    aa = amino_acids['AA'].values
    msasa_dict = {x.upper(): y for x, y in zip(amino_acids['AA3C'], amino_acids['Maximum_SASA'])}

    exp_flag = 'exp205'
    root_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep/{exp_flag}anneal_val_deep'
    protein_sample = pd.read_csv(f'{root_dir}/../sample.csv')
    pdb_selected = protein_sample['pdb'].values

    core_all = np.array([])
    surface_all = np.array([])
    middle_all = np.array([])

    core_cons = np.array([])
    surface_cons = np.array([])
    middle_cons = np.array([])

    num = 100
    for pdb_id in pdb_selected:
        df_beads = pd.read_csv(f'{root_dir}/../{pdb_id}_bead_sasa.csv')
        sasa = df_beads['sasa'].values
        msasa = np.array([msasa_dict[x] for x in df_beads['group_name'].values])
        fsasa = sasa / msasa

        core_idx = (fsasa < 0.10)
        surface_idx = (fsasa > 0.30)
        middle_idx = (fsasa > 0.10) & (fsasa < 0.30)
        seq_best_all = []

        for j in range(num):
            data_anneal = h5py.File(f'{root_dir}/{pdb_id}_profile_{j}.h5', 'r')
            designed_seq = data_anneal['profile'][()]
            seq_best = designed_seq[1]
            seq_best_all.append(seq_best)

            if j == 0:
                seq_native = designed_seq[0]
                seq_native_aa = np.array([idx2aa[x] for x in seq_native])
                print(f'{pdb_id}\n'
                      f'native_core: {seq_native_aa[core_idx]}\n'
                      f'native_middle: {seq_native_aa[middle_idx]}\n'
                      f'native_surface: {seq_native_aa[surface_idx]}\n')
                fig = pl.figure()
                aa_bins = np.arange(21) - 0.5
                pl.hist(seq_native[core_idx], bins=aa_bins, label='core', histtype='step')
                pl.hist(seq_native[surface_idx], bins=aa_bins, label='surface', histtype='step')
                pl.hist(seq_native[middle_idx], bins=aa_bins, label='middle', histtype='step')
                pl.xticks(np.arange(20), labels=aa)
                pl.legend()
                pl.title('native residue use in core / middle / surface')
                pl.savefig(f'{root_dir}/{pdb_id}_native_residue_use_core_surface.pdf')
                pl.close(fig)

        seq_best_all = np.vstack(seq_best_all).T
        conserv_score = get_conservation(seq_best_all)

        core = seq_best_all[core_idx].flatten()
        surface = seq_best_all[surface_idx].flatten()
        middle = seq_best_all[middle_idx].flatten()

        core_all = np.append(core_all, core)
        surface_all = np.append(surface_all, surface)
        middle_all = np.append(middle_all, middle)

        fig = pl.figure()
        aa_bins = np.arange(21) - 0.5
        pl.hist(core, bins=aa_bins, label='core', histtype='step')
        pl.hist(surface, bins=aa_bins, label='surface', histtype='step')
        pl.hist(middle, bins=aa_bins, label='middle', histtype='step')
        pl.xticks(np.arange(20), labels=aa)
        pl.ylabel('N')
        pl.legend()
        pl.title('residue use in core / middle / surface')
        pl.savefig(f'{root_dir}/{pdb_id}_design_residue_use_core_surface.pdf')
        pl.close(fig)

        core_cons = np.append(core_cons, conserv_score[core_idx])
        surface_cons = np.append(surface_cons, conserv_score[surface_idx])
        middle_cons = np.append(middle_cons, conserv_score[middle_idx])

        fig = pl.figure()
        con_bins = np.arange(14) * 0.2 + 0.4
        pl.hist(conserv_score[core_idx], bins=con_bins, label='core', histtype='step')
        pl.hist(conserv_score[surface_idx], bins=con_bins, label='surface', histtype='step')
        pl.hist(conserv_score[middle_idx], bins=con_bins, label='middle', histtype='step')
        pl.legend()
        pl.xlabel('conservation score')
        pl.ylabel('N')
        pl.title('conservation score in core / middle / surface')
        pl.savefig(f'{root_dir}/{pdb_id}_design_conservation_score_core_surface.pdf')
        pl.close(fig)

        fig = pl.figure()
        pl.plot(fsasa, conserv_score, 'bo')
        pl.ylabel('conservation score')
        pl.xlabel('relative SASA')
        pl.savefig(f'{root_dir}/{pdb_id}_design_conservation_score_sasa.pdf')
        pl.close(fig)

    pd.DataFrame({'core': core_all}).to_csv(f'{root_dir}/all_design_core_residues.csv', index=False)
    pd.DataFrame({'middle': middle_all}).to_csv(f'{root_dir}/all_design_middle_residues.csv', index=False)
    pd.DataFrame({'surface': surface_all}).to_csv(f'{root_dir}/all_design_surface_residues.csv', index=False)

    fig = pl.figure()
    aa_bins = np.arange(21) - 0.5
    pl.hist(core_all, bins=aa_bins, label='core', histtype='step')
    pl.hist(surface_all, bins=aa_bins, label='surface', histtype='step')
    pl.hist(middle_all, bins=aa_bins, label='middle', histtype='step')
    pl.xticks(np.arange(20), labels=aa)
    pl.ylabel('N')
    pl.legend()
    pl.title('residue use in core / middle / surface')
    pl.savefig(f'{root_dir}/all_design_residue_use_core_surface.pdf')
    pl.close(fig)

    fig = pl.figure()
    con_bins = np.arange(14) * 0.2 + 0.4
    pl.hist(core_cons, bins=con_bins, label='core', histtype='step')
    pl.hist(surface_cons, bins=con_bins, label='surface', histtype='step')
    pl.hist(middle_cons, bins=con_bins, label='middle', histtype='step')
    pl.legend()
    pl.xlabel('conservation score')
    pl.ylabel('N')
    pl.title('conservation score in core / middle / surface')
    pl.savefig(f'{root_dir}/all_design_conservation_score_core_surface.pdf')
    pl.close(fig)


def plot_sasa_seq_swap():
    amino_acids = pd.read_csv('/home/hyang/bio/erf/data/amino_acids_msasa.csv')
    aa = amino_acids['AA'].values
    msasa_dict = {x.upper(): y for x, y in zip(amino_acids['AA3C'], amino_acids['Maximum_SASA'])}

    exp_flag = 'exp205'
    root_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep/{exp_flag}swap_val_deep'
    protein_sample = pd.read_csv(f'{root_dir}/../sample.csv')
    pdb_selected = protein_sample['pdb'].values

    core_all = np.array([])
    surface_all = np.array([])
    middle_all = np.array([])

    for pdb_id in pdb_selected:
        df_beads = pd.read_csv(f'{root_dir}/../{pdb_id}_bead_sasa.csv')
        sasa = df_beads['sasa'].values
        msasa = np.array([msasa_dict[x] for x in df_beads['group_name'].values])
        fsasa = sasa / msasa
        core_idx = (fsasa < 0.10)
        surface_idx = (fsasa > 0.30)
        middle_idx = (fsasa > 0.10) & (fsasa < 0.30)

        data_anneal = h5py.File(f'{root_dir}/{pdb_id}_profile.h5', 'r')
        designed_seq = data_anneal['profile'][()]
        seq_best = designed_seq[1]

        core = seq_best[core_idx]
        surface = seq_best[surface_idx]
        middle = seq_best[middle_idx]

        core_all = np.append(core_all, core)
        surface_all = np.append(surface_all, surface)
        middle_all = np.append(middle_all, middle)

        fig = pl.figure()
        aa_bins = np.arange(21) - 0.5
        pl.hist(core, bins=aa_bins, label='core', histtype='step')
        pl.hist(surface, bins=aa_bins, label='surface', histtype='step')
        pl.hist(middle, bins=aa_bins, label='middle', histtype='step')
        pl.xticks(np.arange(20), labels=aa)
        pl.legend()
        pl.title('residue use in core / middle / surface')
        pl.savefig(f'{root_dir}/{pdb_id}_swap_residue_use_core_surface.pdf')
        pl.close(fig)

    fig = pl.figure()
    aa_bins = np.arange(21) - 0.5
    pl.hist(core_all, bins=aa_bins, label='core', histtype='step')
    pl.hist(surface_all, bins=aa_bins, label='surface', histtype='step')
    pl.hist(middle_all, bins=aa_bins, label='middle', histtype='step')
    pl.xticks(np.arange(20), labels=aa)
    pl.legend()
    pl.title('residue use in core / middle / surface')
    pl.savefig(f'{root_dir}/all_design_residue_use_core_surface.pdf')
    pl.close(fig)


def analyze_seq_swap():
    amino_acids = pd.read_csv('/home/hyang/bio/erf/data/amino_acids_msasa.csv')
    vocab = {x - 1: y for x, y in zip(amino_acids.idx, amino_acids.AA3C)}
    vocab2 = {x - 1: y for x, y in zip(amino_acids.idx, amino_acids.AA)}

    exp_flag = 'exp205'
    root_dir = f'/home/hyang/bio/erf/data/design/cullpdb_val_deep/{exp_flag}swap_val_deep'
    protein_sample = pd.read_csv(f'{root_dir}/../sample.csv')
    pdb_selected = protein_sample['pdb'].values

    for pdb_id in pdb_selected:
        data_anneal = h5py.File(f'{root_dir}/{pdb_id}_profile.h5', 'r')
        designed_seq = data_anneal['profile'][()]
        seq_native = np.array([vocab[x] for x in designed_seq[0]])
        seq_best = np.array([vocab[x] for x in designed_seq[1]])

        df_beads = pd.read_csv(f'{root_dir}/../{pdb_id}_bead.csv')
        coords = df_beads[['xcb', 'ycb', 'zcb']].values
        num = np.arange(coords.shape[0])
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        for seq, flag in zip([seq_native, seq_best], ['native', 'best']):
            with open(f'{root_dir}/{pdb_id}_{flag}.pdb', 'wt') as mf:
                for i in range(len(num)):
                    mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')

        with open(f'{root_dir}/{pdb_id}_native_best.fa', 'wt') as mf:
            seq_native = ''.join([vocab2[x] for x in designed_seq[0]])
            seq_best = ''.join([vocab2[x] for x in designed_seq[1]])
            mf.write(f'>native\n{seq_native}\n')
            mf.write(f'>best\n{seq_best}\n')


def trrosetta():
    import mdtraj as md
    import pandas as pd
    import numpy as np
    from Bio.PDB import Selection, PDBParser

    amino_acids = pd.read_csv('amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]

    def get_coords(pdb_path):
        p = PDBParser()
        structure = p.get_structure('X', pdb_path)
        residue_list = Selection.unfold_entities(structure, 'R')
        ca_center_list = []
        cb_center_list = []
        for res in residue_list:
            if res.get_resname() not in vocab_aa:
                continue
            try:
                res['CA'].get_coord()
                if res.get_resname() != 'GLY':
                    res['CB'].get_coord()
            except KeyError:
                print(f'{pdb_path}, {res} missing CA / CB atoms')
                continue
            ca_center_list.append(res['CA'].get_coord())
            if res.get_resname() != 'GLY':
                cb_center_list.append(res['CB'].get_coord())
            else:
                cb_center_list.append(res['CA'].get_coord())
        ca_center = np.vstack(ca_center_list)
        cb_center = np.vstack(cb_center_list)
        return ca_center, cb_center

    pdb_list = pd.read_csv('list.txt')['pdb']
    for pdb in pdb_list:
        # if pdb in ['3P0C_A', '4M1X_A', '5ZGM_A', '6H8O_A']:
        #     continue
        # coords_native_ca, coords_native_cb = get_coords(f'{pdb}.pdb')
        df = pd.read_csv(f'{pdb}_bead.csv')
        coords_native_ca = df[['xca', 'yca', 'zca']].values
        coords_native_cb = df[['xcb', 'ycb', 'zcb']].values
        coords_ca_list = [coords_native_ca]
        coords_cb_list = [coords_native_cb]
        flag_list = ['PDB']
        for flag in ['native', 'd1', 'd2']:
            for i in range(5):
                coords_ca, coords_cb = get_coords(f'{pdb}_{flag}_{i}.pdb')
                assert(coords_ca.shape[0] == coords_native_ca.shape[0])
                coords_ca_list.append(coords_ca)
                coords_cb_list.append(coords_cb)
                flag_list.append(f'{flag}_{i}')

        # compute RMSD,
        coords_ca_all = np.stack(coords_ca_list, axis=0)
        coords_cb_all = np.stack(coords_cb_list, axis=0)

        t = md.Trajectory(xyz=coords_ca_all, topology=None)
        t = t.superpose(t, frame=0)
        rmsd_ca = md.rmsd(t, t, frame=0)  # computation will change sample_xyz;

        t = md.Trajectory(xyz=coords_cb_all, topology=None)
        t = t.superpose(t, frame=0)
        rmsd_cb = md.rmsd(t, t, frame=0)  # computation will change sample_xyz;

        # print(pdb, rmsd_cb)
        df = pd.DataFrame({'flag': flag_list, 'rmsd_ca': rmsd_ca, 'rmsd_cb': rmsd_cb})
        df.to_csv(f'{pdb}_rmsd.csv', float_format='%.3f', index=False)

    for pdb in pdb_list:
        df = pd.read_csv(f'{pdb}_rmsd.csv')
        print(pdb)
        print(df)





