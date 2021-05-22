import pandas as pd
import numpy as np
from tqdm import tqdm


# write bead DataFrame to PDB file
# df = pd.read_csv('data/cath/12AS_A_bead.csv')
df = pd.read_csv('data/visual/2HBA_A_bead.csv')

num = df['group_num'].values
name = df['group_name'].values
x, y, z = df['xcb'].values, df['ycb'].values, df['zcb'].values

with open('data/visual/2HBA_A_bead.pdb', 'wt') as mf:
    for i in range(df.shape[0]):
        mf.write(f'ATOM  {num[i]:5d}   CA {name[i]} A{num[i]:4d}    {x[i]:8.2f}{y[i]:8.2f}{z[i]:8.2f}\n')


# write rotated local structure to PDB file
# df = pd.read_csv('data/local/12AS_A.csv')
df = pd.read_csv('data/visual/2HBA_A_CB.csv')
center_num = df['center_num'].unique()

for g in center_num[0:50:5]:
    with open(f'data/visual/2HBA_A_CB_{g}.pdb', 'wt') as mf:
        df_g = df[df['center_num'] == g]
        num = df_g['group_num'].values
        name = df_g['group_name'].values
        x, y, z = df_g['x'].values, df_g['y'].values, df_g['z'].values
        for i in range(df_g.shape[0]):
            mf.write(f'ATOM  {num[i]:5d}   CA {name[i]} A{num[i]:4d}    {x[i]:8.2f}{y[i]:8.2f}{z[i]:8.2f}\n')

# # write rotated local structure to PDB file
# df = pd.read_csv('data/local_rot/12AS_A_rot.csv')
# center_num = df['center_num'].unique()
#
# for g in center_num[:10]:
#     with open(f'data/visual/12AS_A_local_rot_{g}.pdb', 'wt') as mf:
#         df_g = df[df['center_num'] == g]
#         num = df_g['group_num'].values
#         name = df_g['group_name'].values
#         x, y, z = df_g['x'].values, df_g['y'].values, df_g['z'].values
#         for i in range(df_g.shape[0]):
#             mf.write(f'ATOM  {num[i]:5d}   CA {name[i]} A{num[i]:4d}    {x[i]:8.2f}{y[i]:8.2f}{z[i]:8.2f}\n')
#














