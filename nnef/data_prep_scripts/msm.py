import numpy as np
import mdtraj as md
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as pl


"""
prepare the DEShaw trajectories of Fip35 for scoring.
"""


def check_two_trj_are_same():
    # make sure the trj used by Siqin Cao are the same as the trj of I get from DEShaw
    root_dir = '/home/hyang/bio/erf/data/decoys/msm'

    structure = md.load(f'{root_dir}/native_fip35wwdomain.pdb')
    topology = structure.topology

    trj_dir1 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_1-protein/ww_1-protein/'
    trj_dir2 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_2-protein/ww_2-protein/'

    for i, trj_dir in enumerate([trj_dir1, trj_dir2]):
        trj_Cao = md.load(f'{root_dir}/ww_{i+1}-c-alpha.xtc', top=topology)
        coords = trj_Cao.xyz

        structure = md.load(f'{trj_dir}/ww-protein.pdb')
        top = structure.topology
        ca_idx = top.select('name == CA')

        coords_all = []
        flist = pd.read_csv(f'{trj_dir}/flist.txt')['fname']
        for fname in flist:
            trj_deshaw = md.load(f'{trj_dir}/{fname}', top=top)
            coords2 = trj_deshaw.xyz
            coords2_ca = coords2[:, ca_idx, :]
            coords_all.append(coords2_ca)
        coords_all = np.vstack(coords_all)
        dxyz = coords - coords_all
        print(dxyz.max(), dxyz.min())


def extract_cb_index():
    root_dir = '/home/hyang/bio/erf/data/decoys/msm'
    trj_dir1 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_1-protein/ww_1-protein/'
    trj_dir2 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_2-protein/ww_2-protein/'

    for trj_dir in [trj_dir1, trj_dir2]:
        structure = md.load(f'{trj_dir}/ww-protein.pdb')
        top = structure.topology
        ca_gly = top.select('(name == CA) and (resname == GLY)')
        cb = top.select('name == CB')
        beads = np.append(ca_gly, cb)
        beads = np.sort(beads)
        print(beads)
        group_name = [r.name for r in top.residues]
        df = pd.DataFrame({'group_num': np.arange(beads.shape[0]),
                           'group_name': group_name,
                           'beads_cb_index': beads})
        df.to_csv(f'{trj_dir}/ww-protein-beads.csv', index=False)


def write_pdb_sample(seq, coords_sample, data_path):
    with open(data_path, 'wt') as mf:
        for j, coords in enumerate(coords_sample):
            num_steps = (j + 1)
            mf.write('MODEL        '+str(num_steps)+'\n')

            num = np.arange(coords.shape[0])
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            for i in range(len(num)):
                mf.write(f'ATOM  {num[i]:5d}   CA {seq[i]} A{num[i]:4d}    {x[i]:8.3f}{y[i]:8.3f}{z[i]:8.3f}\n')
            mf.write('ENDMDL\n')


def get_fasta():
    amino_acids = pd.read_csv('/home/hyang/bio/erf/data/amino_acids.csv')
    vocab = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    group_name = pd.read_csv('ww-protein-beads.csv')['group_name'].values
    seq = ''.join([vocab[x] for x in group_name])
    with open('seq.fasta', 'wt') as f:
        f.write(seq+'\n')


def get_rmsd():
    root_dir = '/home/hyang/bio/erf/data/decoys/msm'
    trj_dir1 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_1-protein/ww_1-protein/'
    trj_dir2 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_2-protein/ww_2-protein/'
    df = pd.read_csv(f'{root_dir}/fip35_bead.csv')
    coords_native = df[['xcb', 'ycb', 'zcb']].values[None, :, :]

    for i, trj_dir in enumerate([trj_dir1, trj_dir2]):
        structure = md.load(f'{trj_dir}/ww-protein.pdb')
        top = structure.topology
        df = pd.read_csv(f'{trj_dir}/ww-protein-beads.csv')
        cb_idx = df['beads_cb_index'].values
        seq = df['group_name'].values
        state_assignment = pd.read_csv(f'{root_dir}/traj_assigment_0', header=None)[0].values
        state_counts = pd.value_counts(state_assignment)

        coords_list = []
        flist = pd.read_csv(f'{trj_dir}/flist.txt')['fname']
        for k, fname in enumerate(flist):
            trj = md.load(f'{trj_dir}/{fname}', top=top)
            coords_all = trj.xyz * 10
            coords_cb_all = coords_all[:, cb_idx, :]
            coords_list.append(coords_cb_all)
        coords_all = np.vstack(coords_list)
        score = pd.read_csv(f'{trj_dir}/energy.csv')['energy'].values
        # save low energy frames to PDBs
        idx = (score < 1020)
        write_pdb_sample(seq, coords_all[idx], f'{root_dir}/trj{i}_low_energy_sample.pdb')

        # save 10 sample for each state
        coords_sample = []
        t_native = md.Trajectory(xyz=coords_native, topology=None)
        for state, counts in zip(state_counts.index, state_counts.values):
            if counts > 10:
                coords_state = coords_all[(state_assignment == state)]
                rand_idx = np.random.randint(0, coords_state.shape[0], 10)
                coords_sample.append(coords_state[rand_idx])
                # # compute rmsd to native and within states
                t = md.Trajectory(xyz=coords_state.copy(), topology=None)
                rmsd = md.rmsd(t, t_native, frame=0)
                # rmsd = md.rmsd(t, t, frame=0)
                print(rmsd.mean(), rmsd.std())

        coords_sample = np.vstack(coords_sample)
        # print(coords_sample.shape)
        write_pdb_sample(seq, coords_sample, f'{root_dir}/trj{i}_states_sample.pdb')

        score = np.append(np.array([0]), score)
        coords_list = [coords_native] + coords_list
        coords_all = np.vstack(coords_list)

        t = md.Trajectory(xyz=coords_all, topology=None)
        t = t.superpose(t, frame=0)
        rmsd = md.rmsd(t, t, frame=0)
        df = pd.DataFrame({'energy': score, 'rmsd': rmsd})
        df.to_csv(f'{trj_dir}/energy_rmsd.csv', index=False)


def compare_free_energy():
    root_dir = '/home/hyang/bio/erf/data/decoys/msm'
    trj_dir1 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_1-protein/ww_1-protein/'
    trj_dir2 = f'{root_dir}/deshaw/DESRES-Trajectory-ww_2-protein/ww_2-protein/'

    for i, trj_dir in enumerate([trj_dir1, trj_dir2]):
        energy = pd.read_csv(f'{trj_dir}/energy.csv')['energy'].values
        state_assignment = pd.read_csv(f'{root_dir}/traj_assigment_{i}', header=None)[0].values
        state_counts = pd.value_counts(state_assignment)
        num_states = state_counts.shape[0]
        state_energy = np.zeros(num_states)
        state_energy_std = np.zeros(num_states)
        state_free_energy = -np.log(state_counts / state_assignment.shape[0])

        for k, state in enumerate(state_counts.index):
            idx = (state_assignment == state)
            state_energy_k = np.mean(energy[idx])
            print(state, state_energy_k)
            state_energy[k] = state_energy_k
            state_energy_std[k] = np.std(energy[idx])

        fig = pl.figure()
        pl.plot(state_free_energy, state_energy, 'bo')
        pl.xlabel('state free energy')
        pl.ylabel('state energy')
        pl.savefig(f'{root_dir}/trj_free_energy_{i}.pdf')
        pl.close(fig)

        fig = pl.figure()
        pl.errorbar(state_free_energy, state_energy, yerr=state_energy_std, fmt='bo')
        pl.xlabel('state free energy')
        pl.ylabel('state energy')
        pl.savefig(f'{root_dir}/trj_free_energy_{i}_std.pdf')
        pl.close(fig)

    for i, trj_dir in enumerate([trj_dir1, trj_dir2]):
        df = pd.read_csv(f'{trj_dir}/energy_rmsd.csv')
        energy = df['energy'].values
        rmsd = df['rmsd'].values
        # plot RMSD vs. energy
        fig = pl.figure()
        pl.plot(rmsd, energy, 'b.', markersize=0.005)
        pl.xlabel('RMSD')
        pl.ylabel('energy')
        pl.savefig(f'{root_dir}/trj_rmsd_energy_{i}.jpg')
        pl.close(fig)

    for i, trj_dir in enumerate([trj_dir1, trj_dir2]):
        df = pd.read_csv(f'{trj_dir}/energy_rmsd.csv')
        energy = df['energy'].values
        rmsd = df['rmsd'].values
        # plot RMSD vs. Time, Energy vs. Time
        fig = pl.figure()
        pl.subplot(211)
        pl.plot(rmsd, 'g.', markersize=0.005)
        pl.ylabel('RMSD')
        pl.subplot(212)
        pl.plot(energy, 'g.', markersize=0.005)
        pl.ylabel('energy score')
        pl.xlabel('time-steps')
        pl.savefig(f'{root_dir}/trj_rmsd_energy_time_{i}.jpg')
        pl.close(fig)


















