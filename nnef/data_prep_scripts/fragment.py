import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import torch
from physics.protein_os import Protein
import os

"""
This script prepared fragments for sampling with fragment replacement. 

1) generate k-mer fragment with Rosetta
2) the Rosetta k-mer coordinates include only backbone atoms. extract the PDB id of k-mer and match their CB. 

"""


def get_rosetta_vall_pdbs():
    # get the PDB and chain ids used in the Rosetta fragment picker database.
    pdb_list = []
    chain_list = []
    with open('vall.jul19.2011.blast', 'rt') as f:
        for line in f.readlines():
            if line.startswith('>'):
                pdb_chain = line.split(' ')[0][1:]
                pdb_list.append(pdb_chain[0:4].upper())
                chain_list.append(pdb_chain[4:])

    df = pd.DataFrame({'pdb': pdb_list, 'chain': chain_list})
    df.to_csv('rosetta.csv', index=False)


def get_vall_int():
    # in the beginning, I want to calculate CB from CA and torsional angles.
    pdb_list = pd.read_csv('data/fragment/rosetta/flist.txt')['pdb'].values
    mode = 'CB'
    args = None
    device = torch.device('cuda')
    data_path = 'data/fragment/rosetta/'

    used_pdb_list = []
    with h5py.File(f'data/fragment/rosetta_{mode}_int.h5', 'w') as f:
        for pdb_id in tqdm(pdb_list):
            # TODO: this only works for chains with no missing residues
            # seq, coords, profile = load_protein(data_path, pdb_id, mode, device, args)
            # protein = Protein(seq, coords, profile)
            # coords_int = protein.cartesian_to_internal(coords)
            pass


def convert_rosetta_frags(query_pdb='2jsvX'):
    # get PDB id of rosetta fragment, and extract the CB of the fragment;
    # Sometimes this doesn't work, because the position_id used by Rosetta is different from ids used in my PDB files.
    frag_len = 5
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    data_path = f'data/fragment/{query_pdb}/frags.fsc.200.5mers'
    if not os.path.exists(data_path+'.csv'):
        with open(data_path, 'rt') as f:
            with open(data_path+'.csv', 'wt') as f2:
                line = f.readline()
                f2.write(line[1:])
                for line in f.readlines():
                    if not line.startswith('#'):
                        f2.write(line)

    df = pd.read_csv(data_path+'.csv', sep='\s+')
    pdb_id = df['pdbid'].apply(lambda x: x.upper()).values
    chain = df['c'].values
    pdb_pos = df['vall_pos'].values
    query_pos = df['query_pos'].values
    score_diff = np.round(df['SequenceIdentity'].values * frag_len)

    query_fasta = f'data/fragment/{query_pdb}/{query_pdb}.fasta'
    with open(query_fasta, 'rt') as f:
        f.readline()
        query_seq = np.array(list(f.readline()[:-1]))

    ind = np.zeros(df.shape[0], dtype=np.int)
    q_pos_all = []
    coords_int_all = []

    for i in tqdm(range(pdb_id.shape[0])):
        if query_pos[i] <= 2:
            continue
        beads_path = f'data/fragment/prep_frag/rosetta/{pdb_id[i]}_{chain[i]}_bead.csv'
        if not os.path.exists(beads_path):
            continue
        df_beads = pd.read_csv(beads_path)
        idx = np.arange(df_beads.shape[0])
        # gnum = df_beads['group_num_pdb'].values
        # k = idx[gnum == pdb_pos[i]][0]
        gnum = df_beads['group_num'].values
        k = idx[gnum + 1 == pdb_pos[i]]
        if len(k) == 1:
            k = k[0]
        else:
            continue

        if (k < 2) | (k >= df_beads.shape[0]-frag_len):
            # No -2/ or +1 residues of the fragment
            continue
        if np.sum(gnum[k-1:k+frag_len+1] - gnum[k-2:k+frag_len]) != frag_len+2:
            # missing residues
            continue

        seq = df_beads['group_name'].apply(lambda x: vocab[x]).values
        seq_frag = seq[k:k+frag_len]
        seq_q = query_seq[query_pos[i]-1:query_pos[i]+frag_len-1]

        if np.sum(seq_frag != seq_q) != score_diff[i]:
            print('seq not match', pdb_id[i], seq_frag, seq_q)
            continue
        assert(np.sum(seq_frag != seq_q) == score_diff[i])

        # coords of [-2, -1, frag, +1] residues
        coords = df_beads[['xcb', 'ycb', 'zcb']].values[k-2:k+frag_len+1]
        coords = torch.tensor(coords)
        protein = Protein(None, coords, None)
        coords_int = protein.cartesian_to_internal(coords)

        q_pos_all.append(query_pos[i] - 3)  # rosetta query pos starts from 1, -3 converts it to internal index
        coords_int_all.append(coords_int)
        ind[i] = 1

    df[ind == 1].to_csv(data_path+'_int.csv', index=False)
    q_pos_all = np.array(q_pos_all)  # (num_frag,)
    coords_int_all = torch.stack(coords_int_all, dim=0).numpy()  # (num_frag, frag_len, 3)
    print(f'{query_pdb} total number of frags: {q_pos_all.shape[0]}')
    with h5py.File(f'data/fragment/{query_pdb}/{query_pdb}_int.h5', 'w') as f:
        dset = f.create_dataset("query_pos", shape=q_pos_all.shape, data=q_pos_all, dtype='i')
        dset = f.create_dataset("coords_int", shape=coords_int_all.shape, data=coords_int_all, dtype='f4')


def convert_frag_val_deep():
    data_path = 'data/fold/cullpdb_val_deep'
    protein_sample = pd.read_csv(f'{data_path}/sample.csv')
    pdb_selected = protein_sample['pdb'].values
    for pdb_id in pdb_selected[1:]:
        convert_rosetta_frags(pdb_id)


def prep_frag_val_deep():
    # prepare scripts to run Rosetta fragment picker
    data_path = 'data/fold/cullpdb_val_deep'
    protein_sample = pd.read_csv(f'{data_path}/sample.csv')
    pdb_selected = protein_sample['pdb'].values
    amino_acids = pd.read_csv('data/amino_acids.csv')
    vocab = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    with open(f'data/fragment/run_val_deep.sh', 'wt') as fsh:
        for pdb_id in pdb_selected:
            print(pdb_id)
            df_beads = pd.read_csv(f'{data_path}/{pdb_id}_bead.csv')
            seq = ''.join(list(df_beads['group_name'].apply(lambda x: vocab[x]).values))
            frag_path = f'data/fragment/{pdb_id}'
            if not os.path.exists(frag_path):
                os.mkdir(frag_path)
            fsh.write(f'fragment_picker.static.linuxgccrelease @{pdb_id}/flags\n')
            with open(f'{frag_path}/{pdb_id}.fasta', 'wt') as f:
                f.write(f'>{pdb_id}\n{seq}\n')
            with open(f'{frag_path}/flags', 'wt') as f:
                f.write(f"""
# Input databases
-in::file::vall /home/hyang/Software/rosetta/main/tools/fragment_tools/vall.jul19.2011.gz

# Query-related input files
-in::file::fasta                {pdb_id}/{pdb_id}.fasta

# Weights file
-frags::scoring::config         simple.wghts

# What should we do?
-frags::bounded_protocol

# three-mers only, please
-frags::frag_sizes              5
-frags::n_candidates            200
-frags::n_frags                 200

# Output 
-out::file::frag_prefix         {pdb_id}/frags
-frags::describe_fragments      {pdb_id}/frags.fsc
""")








