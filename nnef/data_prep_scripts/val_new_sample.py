import pandas as pd
import numpy as np
from tqdm import tqdm


def prepare_pdb_no_missing_residues():
    # use the hhsuite_CB_cullpdb_val.csv data, and select PDBs with no missing residues.
    pdb_list = pd.read_csv('data/hhsuite_CB_cullpdb_val.csv')['pdb'].values
    no_missing_res_list = []
    seq_len_list = []
    for pdb in tqdm(pdb_list):
        df = pd.read_csv(f'../hhsuite/hhsuite_beads/hhsuite/{pdb}_bead.csv')
        group_num_pdb = df['group_num_pdb'].values
        group_num = df['group_num'].values
        if type(group_num_pdb[0]) == str:
            continue
        num_res_pdb = group_num_pdb[-1] - group_num_pdb[0] + 1
        if group_num.shape[0] == num_res_pdb:
            no_missing_res_list.append(pdb)
            seq_len_list.append(group_num.shape[0])
    df = pd.DataFrame({'pdb': no_missing_res_list, 'seq_len': seq_len_list})
    df.to_csv('data/hhsuite_CB_cullpdb_val_no_missing_residue.csv', index=False)
    df2 = df.sample(100)
    df2.to_csv('data/decoys_seq/hhsuite_CB_cullpdb_val_no_missing_residue_sample.csv', index=False)


# prepare a new design sample; require it to have a evolutionary profile in proteinnet.
bead_dir = '~/bio/proteinnet/text_based/casp12/proteinnet_beads'
profile_dir = '~/bio/proteinnet/text_based/casp12/training_100_profile'
target_dir = '~/bio/erf/data/design/cullpdb_val_sample'
pdb_list = pd.read_csv(f'{target_dir}/hhsuite_CB_cullpdb_val_no_missing_residue_sample.csv')['pdb']

with open(f'cp_sample.sh', 'w') as mf:
    for pdb in pdb_list:
        pdb_id, chain = pdb.split('_')
        mf.write(f'cp {bead_dir}/{pdb}_bead.csv {target_dir}/\n')
        mf.write(f'cp {profile_dir}/{pdb_id}_*_{chain}_profile.csv {target_dir}/{pdb_id}_{chain}_profile.csv\n')

# prepare a new fold sample; a sub sample of the new design sample.
pdb_list = pd.read_csv(f'sample.csv')['pdb']
seq_len = []
target_dir = '~/bio/erf/data/fold/cullpdb_val_sample/'
for pdb in pdb_list:
    df = pd.read_csv(f'{pdb}_bead.csv')
    if df.shape[0] <= 80:
        # print(pdb, df.shape[0])
        print(f'cp {pdb}_bead.csv {target_dir}')
        print(f'cp {pdb}_profile.csv {target_dir}')

    seq_len.append(df.shape[0])
seq_len = np.array(seq_len)
idx = (seq_len < 100)

# rename old protein sample from proteinnet
pdb_list = pd.read_csv(f'sample.csv')['pdb_id']
with open(f'cp_sample.sh', 'w') as mf:
    for pdb_id in pdb_list:
        pdb, _, chain = pdb_id.split('_')
        mf.write(f'cp {pdb_id}_profile.csv {pdb}_{chain}_profile.csv\n')

# prepare a new small sample for deep folding / design

df = pd.read_csv('data/hhsuite_CB_cullpdb_val_no_missing_residue.csv')
pdb_id = df['pdb']
chain = pdb_id.apply(lambda x: x.split('_')[1])
seq_len = df['seq_len'].values
idx = (seq_len < 120) & (seq_len > 50) & (chain == 'A')
df[idx].to_csv('data/hhsuite_CB_cullpdb_val_no_missing_residue_deep_50-120.csv', index=False)


