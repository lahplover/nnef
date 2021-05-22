import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.PDB import Selection, PDBParser
import os


def extract_beads(pdb_path):
    amino_acids = pd.read_csv('/home/hyang/bio/erf/data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    try:
        structure = p.get_structure('X', pdb_path)
    except:
        return 0
    residue_list = Selection.unfold_entities(structure, 'R')

    ca_center_list = []
    cb_center_list = []
    res_name_list = []
    res_num_list = []
    chain_list = []

    for res in residue_list:
        if res.get_resname() not in vocab_aa:
            # raise ValueError('protein has non natural amino acids')
            continue
        chain_list.append(res.parent.id)
        res_name_list.append(vocab_dict[res.get_resname()])
        res_num_list.append(res.id[1])
        try:
            ca_center_list.append(res['CA'].get_coord())
        except KeyError:
            return 0
        if res.get_resname() != 'GLY':
            try:
                cb_center_list.append(res['CB'].get_coord())
            except KeyError:
                return 0
        else:
            cb_center_list.append(res['CA'].get_coord())

    ca_center = np.vstack(ca_center_list)
    cb_center = np.vstack(cb_center_list)

    df = pd.DataFrame({'chain_id': chain_list,
                       'group_num': res_num_list,
                       'group_name': res_name_list,
                       'x': ca_center[:, 0],
                       'y': ca_center[:, 1],
                       'z': ca_center[:, 2],
                       'xcb': cb_center[:, 0],
                       'ycb': cb_center[:, 1],
                       'zcb': cb_center[:, 2]})

    df.to_csv(f'{pdb_path}_bead.csv', index=False)
    return 1


def extract_md():
    pdb_id_list = pd.read_csv('list', header=None, names=['pdb'])['pdb'].values

    for pdb_id in pdb_id_list:
        root_dir = f'/home/hyang/bio/openmm/data/{pdb_id}/decoy'
        df = pd.read_csv(f'{root_dir}/list.csv')
        pdb_list = df['pdb'].values
        # add the initial PDBs before MD run.
        pdb_list = np.append(np.array([f'solvated_EM_protein.pdb']), pdb_list)
        bad_list = []
        for i, pdb in tqdm(enumerate(pdb_list)):
            pdb_path = f'{root_dir}/{pdb}'
            # if os.path.exists(pdb_path + '_bead.csv'):
            #     continue
            result = extract_beads(pdb_path)
            if result == 0:
                # some structure prediction only has CA.
                bad_list.append(pdb)
                pdb_list[i] = '0'
        if len(bad_list) > 0:
            pdb_list = pdb_list[pdb_list != '0']
        # add the native structure to pdb_list
        pdb_list = np.append(np.array([f'{pdb_id}_A']), pdb_list)
        df = pd.DataFrame({'pdb': pdb_list})
        df.to_csv(f'{root_dir}/flist.csv', index=False)


def rmsd_md():
    # calculate rmsd using mdtraj
    import mdtraj as md

    pdb_id_list = pd.read_csv('list', header=None, names=['pdb'])['pdb'].values

    for pdb_id in tqdm(pdb_id_list):
        root_dir = f'/home/hyang/bio/openmm/data/{pdb_id}/decoy'
        df = pd.read_csv(f'{root_dir}/flist.csv')
        pdb_list = df['pdb'].values

        xyz_list = []
        for pdb in pdb_list:
            xyz = pd.read_csv(f'{root_dir}/{pdb}_bead.csv')[['xcb', 'ycb', 'zcb']].values
            xyz_list.append(xyz[None, :, :])
        xyz_all = np.vstack(xyz_list)
        traj = md.Trajectory(xyz=xyz_all, topology=None)
        rmsd = md.rmsd(traj, traj, 0)
        df = pd.DataFrame({'pdb': pdb_list, 'rmsd': rmsd})
        df.to_csv(f'{root_dir}/flist_rmsd.csv', index=False)









