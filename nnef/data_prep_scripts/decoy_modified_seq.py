import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp


def prepare_decoys_modified_seq():
    amino_acids = pd.read_csv('data/amino_acids.csv')
    aa = amino_acids.AA.values
    vocab = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}

    aa_types = pd.read_csv('data/aa_types.csv')
    aa_type2 = {x: y for x, y in zip(aa_types.AA, aa_types[f'type2'])}
    aa_type9 = {x: y for x, y in zip(aa_types.AA, aa_types[f'type9'])}

    pdb_list = pd.read_csv('data/decoys/decoys_seq/hhsuite_CB_cullpdb_val_no_missing_residue_sample.csv')['pdb'].values
    for pdb in tqdm(pdb_list):
        df = pd.read_csv(f'../hhsuite/hhsuite_beads/hhsuite/{pdb}_bead.csv')
        seq = df['group_name'].apply(lambda x: vocab[x]).values
        num = 100

        # make random sequences
        seq_random = []
        for i in range(num):
            sid = np.random.randint(0, 20, len(seq))
            s = ''.join(aa[sid])
            seq_random.append(s)
        df_seq = pd.DataFrame({'seq': seq_random})
        df_seq.to_csv(f'data/decoys/decoys_seq/{pdb}_seq_random.csv', index=False)

        # make shuffle sequences
        seq_shuffle = []
        s = seq.copy()
        for i in range(num):
            np.random.shuffle(s)
            seq_shuffle.append(''.join(s))
        df_seq = pd.DataFrame({'seq': seq_shuffle})
        df_seq.to_csv(f'data/decoys/decoys_seq/{pdb}_seq_shuffle.csv', index=False)

        # make sequences all Leu
        # make polar-nonpolar 2 classes replacements
        seq_type2 = []
        for i in range(num):
            s = seq.copy()
            for j in range(len(s)):
                c_type = aa_type2[s[j]]  # c_type = 0 / 1, polar / non-polar classification
                c_aa_type = aa[aa_types['type2'] == c_type]
                s[j] = np.random.choice(c_aa_type)
            seq_type2.append(''.join(s))
        df_seq = pd.DataFrame({'seq': seq_type2})
        df_seq.to_csv(f'data/decoys/decoys_seq/{pdb}_seq_type2.csv', index=False)

        # make polar-nonpolar 2 classes replacements, polar -> D, nonpolar -> L
        seq_type2LD = []
        for i in range(num):
            s = seq.copy()
            for j in range(len(s)):
                c_type = aa_type2[s[j]]  # c_type = 0 / 1, polar / non-polar classification
                if c_type == 0:
                    s[j] = 'L'
                else:
                    s[j] = 'D'
            seq_type2LD.append(''.join(s))
        df_seq = pd.DataFrame({'seq': seq_type2LD})
        df_seq.to_csv(f'data/decoys/decoys_seq/{pdb}_seq_type2LD.csv', index=False)

        # make 9 classes replacements
        seq_type9 = []
        for i in range(num):
            s = seq.copy()
            for j in range(len(s)):
                c_type = aa_type9[s[j]]  # c_type = 9 classes classification
                c_aa_type = aa[aa_types['type9'] == c_type]
                s[j] = np.random.choice(c_aa_type)
            seq_type9.append(''.join(s))
        df_seq = pd.DataFrame({'seq': seq_type9})
        df_seq.to_csv(f'data/decoys/decoys_seq/{pdb}_seq_type9.csv', index=False)


if __name__ == '__main__':
    prepare_decoys_modified_seq()





