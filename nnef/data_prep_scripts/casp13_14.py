import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.PDB import Selection, PDBParser

"""
This script is to extract beads from the predicted structures in CASP13 and CASP14 after the competitions.
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


def extract_casp13_14():
    # casp_id = 'casp13'
    casp_id = 'casp14'
    root_dir = f'/home/hyang/bio/erf/data/decoys/{casp_id}/'
    casp = pd.read_csv(f'{root_dir}/pdb_list.txt')['pdb'].values
    modified_casp_id = []
    for casp_id in tqdm(casp):
        pdb_list = pd.read_csv(f'{root_dir}/{casp_id}/flist.txt')['pdb'].values
        ca_only_list = []
        for i, pdb_id in enumerate(pdb_list):
            pdb_path = f'{root_dir}/{casp_id}/{pdb_id}'
            result = extract_beads(pdb_path)
            if result == 0:
                # some structure prediction only has CA.
                ca_only_list.append(pdb_id)
                pdb_list[i] = '0'
        if len(ca_only_list) > 0:
            pdb_list = pdb_list[pdb_list != '0']
            df = pd.DataFrame({'pdb': pdb_list})
            df.to_csv(f'{root_dir}/{casp_id}/flist.txt', index=False)
            modified_casp_id.append(casp_id)


def check_residue_num():
    # some groups submit models for only parts of the domains, exclude those models.
    casp_id = 'casp14'
    root_dir = f'/home/hyang/bio/erf/data/decoys/{casp_id}/'
    casp = pd.read_csv(f'{root_dir}/pdb_list.txt')['pdb'].values
    for casp_id in tqdm(casp):
        pdb_list = pd.read_csv(f'{root_dir}/{casp_id}/flist.txt')['pdb'].values
        num = np.zeros(len(pdb_list))
        for i, pdb_id in enumerate(pdb_list):
            df = pd.read_csv(f'{root_dir}/{casp_id}/{pdb_id}_bead.csv')
            num[i] = df.shape[0]
        if len(np.unique(num)) > 1:
            seq_len = np.median(num)
            pdb_list = pdb_list[(num == seq_len)]
            df = pd.DataFrame({'pdb': pdb_list})
            df.to_csv(f'{root_dir}/{casp_id}/flist.txt', index=False)
            print(casp_id, seq_len, num)


def check_missing_residues():
    # chech which casp14 evaluation units have gaps
    casp_id = 'casp14'
    root_dir = f'/home/hyang/bio/erf/data/decoys/{casp_id}/'
    casp = pd.read_csv(f'{root_dir}/pdb_list.txt')['pdb'].values

    no_missing_res_list = []
    seq_len_list = []
    idx = np.zeros(casp.shape[0])
    for i, pdb in tqdm(enumerate(casp)):
        pdb_list = pd.read_csv(f'{root_dir}/{pdb}/flist.txt')['pdb'].values
        pdb_id = pdb_list[0]
        df_i = pd.read_csv(f'{root_dir}/{pdb}/{pdb_id}_bead.csv')
        group_num = df_i['group_num'].values
        num_res_pdb = group_num[-1] - group_num[0] + 1
        if group_num.shape[0] == num_res_pdb:
            no_missing_res_list.append(pdb)
            seq_len_list.append(group_num.shape[0])
            idx[i] = 1
    casp_missing_residues = casp[idx == 0].copy()
    df = pd.DataFrame({'pdb': casp_missing_residues})
    df.to_csv(f'{root_dir}/{casp_id}_missing_residues.csv', index=False)


