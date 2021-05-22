import pandas as pd
import numpy as np
from Bio.PDB import Selection, PDBParser
import os

"""
convert Robetta predicted structures into beads
"""


def extract_beads(pdb_path):
    amino_acids = pd.read_csv('/home/hyang/bio/erf/data/amino_acids.csv')
    vocab_aa = [x.upper() for x in amino_acids.AA3C]
    vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    p = PDBParser()
    structure = p.get_structure('X', pdb_path)
    i = 1
    for model in structure.child_list:
        residue_list = Selection.unfold_entities(model, 'R')

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

        df.to_csv(f'{pdb_path}_{i}_bead.csv', index=False)
        i += 1


root_dir = '/home/hyang/bio/erf/data/design/cullpdb_val_deep/'
pdb_list = pd.read_csv(f'{root_dir}/sample.csv')['pdb'].values

for pdb in pdb_list:
    for flag in ['d1', 'd2']:
        extract_beads(f'{root_dir}/exp205robetta/{pdb}_{flag}.pdb')

for pdb in pdb_list:
    for flag in ['d1', 'd2']:
        # write designed sequence in native structure
        df = pd.read_csv(f'{root_dir}/{pdb}_bead.csv')
        df2 = pd.read_csv(f'{root_dir}/exp205robetta/{pdb}_{flag}.pdb_{1}_bead.csv')
        df['group_name'] = df2['group_name']
        df.to_csv(f'{root_dir}/exp205robetta/{pdb}_{flag}.pdb_native_bead.csv', index=False)



