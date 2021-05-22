import pandas as pd
import numpy as np
from Bio.PDB import Selection, PDBParser
import multiprocessing as mp
import os

"""
convert Rosetta ab-initio folding predicted structures into beads; only for 1ZZK
"""


def extract_beads(pdb_path):
    amino_acids = pd.read_csv('/home/hyang/bio/erf/data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', pdb_path)
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

        try:
            res['CA'].get_coord()
            if res.get_resname() != 'GLY':
                res['CB'].get_coord()
        except KeyError:
            print(f'{pdb_path}, {res} missing CA / CB atoms')
            continue

        chain_list.append(res.parent.id)
        res_name_list.append(vocab_dict[res.get_resname()])
        res_num_list.append(res.id[1])

        ca_center_list.append(res['CA'].get_coord())
        if res.get_resname() != 'GLY':
            cb_center_list.append(res['CB'].get_coord())
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


root_dir = '/home/hyang/bio/erf/data/rosetta/1ZZK_A/pdbs/'
pdb_list = pd.read_csv(f'{root_dir}/list.txt')['pdb'].values


def extract_batch(batch):
    for pdb in batch:
        if not os.path.exists(f'{root_dir}/{pdb}_bead.csv'):
            extract_beads(f'{root_dir}/{pdb}')


N = len(pdb_list)
num_cores = 100
batch_size = N // num_cores + 1

batch_list = []
for i in range(0, N, batch_size):
    batch = pdb_list[i:i+batch_size]
    batch_list += [batch]

with mp.Pool(processes=num_cores) as pool:
    pool.map(extract_batch, batch_list)





