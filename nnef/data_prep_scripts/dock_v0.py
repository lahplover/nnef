import pandas as pd
from Bio.PDB import Selection, PDBParser
# from Bio.PDB.vectors import rotmat, Vector
import numpy as np


"""
PDB file --> beads center DataFrame --> local structure --> rotated local structure 
Functions in this version can handle multiple chains PDB file. 
"""


def get_bead_center(residue):
    # return CA coordinates if residue is GLY
    ca_coord = residue['CA'].get_coord()
    if residue.get_resname() == "GLY":
        return ca_coord
    # for other residues, return mean of CA coord and side chain mass centroid
    backbone_atoms = {'N', 'CA', 'C', 'O'}
    atom_mass = {'C': 12.0, 'N': 14.0, 'O': 16.0, 'S': 32.0, 'H': 1.0}

    weighted_coord = np.array([0.0, 0.0, 0.0])
    total_weight = 0

    for atom in residue.get_atoms():
        if atom.get_name() not in backbone_atoms:
            weight = atom_mass[atom.element]
            coord = atom.get_coord()
            weighted_coord += weight * coord
            total_weight += weight

    side_chain_center = weighted_coord / total_weight
    bead_center = (ca_coord + side_chain_center) / 2

    return bead_center


def extract_beads(pdb_file):
    """
    convert PDB to pandas dataframe
    :param pdb_file:
    :return:
    """
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]

    p = PDBParser()
    structure = p.get_structure('X', f'data/dock/pdb/{pdb_file}.pdb')
    residue_list = Selection.unfold_entities(structure, 'R')

    bead_center_list = []
    res_name_list = []
    res_num_list = []
    chain_list = []

    for res in residue_list:
        if res.get_resname() not in vocab_aa:
            # raise ValueError('protein has non natural amino acids')
            continue
        chain_list.append(res.parent.id)
        res_name_list.append(res.get_resname())
        res_num_list.append(res.id[1])
        bead_center = get_bead_center(res)
        bead_center_list.append(bead_center)

    g_center = np.vstack(bead_center_list)

    df = pd.DataFrame({'chain_id': chain_list,
                       'group_num': res_num_list,
                       'group_name': res_name_list,
                       'x': g_center[:, 0],
                       'y': g_center[:, 1],
                       'z': g_center[:, 2]})

    df.to_csv(f'data/dock/beads/{pdb_file}_bead.csv', index=False)


def _rotation_matrix(c1, c2):
    z = np.cross(c1, c2)
    x = c1
    y = np.cross(z, x)
    x = x / np.sqrt(np.sum(x ** 2))
    y = y / np.sqrt(np.sum(y ** 2))
    z = z / np.sqrt(np.sum(z ** 2))
    R = np.vstack([x, y, z])
    # Rinv = np.linalg.inv(R.T)
    return R


def rotate_one(fname):
    df_list = []
    df = pd.read_csv(f'data/dock/local/{fname}.csv')
    center_num = df['center_num'].unique()
    for g in center_num:
        df_g = df[df['center_num'] == g]

        df_g = df_g.sort_values(by=['chain_id', 'group_num'])
        g_chain = df_g['chain_id'].values
        g_group_num = df_g['group_num'].values
        n_res = df_g.shape[0]
        idx = np.arange(n_res)
        # locate the central residue
        center_idx = (g_chain == g[0]) & (g_group_num == int(g[1:]))
        i = idx[center_idx][0]
        if (i == 0) | (i == n_res-1):
            # i is the smallest or largest, so can't find previous or next group for the coordinates calculation
            # continue
            raise ValueError('can not find previous or next group')
        # make sure the previous and next residues are from the same chain
        assert(g_chain[i-1] == g[0])
        assert(g_chain[i+1] == g[0])

        coords = df_g[['x', 'y', 'z']].values
        coords = coords - coords[i]  # center
        # coords of the previous and next group in local peptide
        c1 = coords[i-1]
        c2 = coords[i+1]

        rotate_mat = _rotation_matrix(c1, c2)

        coords = np.squeeze(np.matmul(rotate_mat[None, :, :], coords[:, :, None]))

        distance = np.sqrt(np.sum(coords**2, axis=1))

        segment_info = np.ones(n_res, dtype=int) * 3
        segment_info[i] = 0
        segment_info[i-1] = 1
        segment_info[i+1] = 2

        df_g = pd.DataFrame({'center_num': df_g['center_num'],
                             'chain_id': df_g['chain_id'],
                             'group_num': df_g['group_num'],
                             'group_name': df_g['group_name'],
                             'x': coords[:, 0],
                             'y': coords[:, 1],
                             'z': coords[:, 2],
                             'distance': distance,
                             'segment': segment_info})
        df_g = df_g.sort_values(by='distance')
        df_list.append(df_g)
    df = pd.concat(df_list, ignore_index=True)
    df['num'] = np.arange(center_num.shape[0]).repeat(10, axis=0)
    df.to_csv(f'data/dock/local_rot/{fname}_rot.csv', index=False)


def extract_one_topk(fname, k=10):
    df = pd.read_csv(f'data/dock/beads/{fname}_bead.csv', dtype={'chain_id': str, 'group_num': int})
    # record the positions of chain start and end
    unique_chains = df['chain_id'].unique()
    chain = df['chain_id'].values
    chain_start_end = [0]
    for c in unique_chains:
        chain_len = chain[chain == c].shape[0]
        chain_start_end += [chain_start_end[-1]+chain_len-1, chain_start_end[-1]+chain_len]
    chain_start_end = chain_start_end[:-1]
    # print(chain_start_end)

    gnum = df['group_num'].values
    # gnum = np.array([c + g for c, g in zip(df['chain_id'].values, df['group_num'].values)])
    gname = df['group_name'].values
    gcoords = df[['x', 'y', 'z']].values

    dist_mat = np.sqrt(((gcoords[None, :, :] - gcoords[:, None, :]) ** 2).sum(axis=2))

    chain_id_list = []
    gc_num_list = []
    g_num_list = []
    g_name_list = []
    coords_list = []
    dist_list = []

    for i, gc in enumerate(gnum):
        if i in chain_start_end:
            continue
        dist_i = dist_mat[i]
        dist_i_arg = np.argsort(dist_i)
        topk_arg = dist_i_arg[:k]
        gnum_i = gnum[topk_arg]
        gname_i = gname[topk_arg]
        chain_i = chain[topk_arg]
        coords_i = gcoords[topk_arg]
        dist_i = dist_i[topk_arg]

        g_num_list.extend(gnum_i)
        gc_num_list.extend([chain[i] + str(gc)] * len(gnum_i))
        g_name_list.extend(gname_i)
        chain_id_list.extend(chain_i)
        coords_list.append(coords_i)
        dist_list.extend(dist_i)

    coords = np.vstack(coords_list)
    df_chain = pd.DataFrame({'center_num': gc_num_list,
                             'chain_id': chain_id_list,
                             'group_num': g_num_list,
                             'group_name': g_name_list,
                             'x': coords[:, 0],
                             'y': coords[:, 1],
                             'z': coords[:, 2],
                             'distance': dist_list})

    df_chain.to_csv(f'data/dock/local/{fname}.csv', index=False)


if __name__ == '__main__':
    pdb_list = pd.read_csv('data/dock/pdb_list.txt')['fname'].values
    for pdb_id in pdb_list:
        extract_beads(pdb_id)
        # convert the structure to a list of local structure
        extract_one_topk(pdb_id)
        rotate_one(pdb_id)




