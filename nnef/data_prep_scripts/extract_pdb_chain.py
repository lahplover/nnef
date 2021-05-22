import pandas as pd
from Bio.PDB import Select, PDBParser, PDBList, PDBIO
from Bio import BiopythonWarning
# from Bio.PDB.vectors import rotmat, Vector
import numpy as np
from tqdm import tqdm
import os
import warnings


amino_acids = pd.read_csv('data/amino_acids.csv')
vocab_aa = [x.upper() for x in amino_acids.AA3C]
vocab_dict = {x.upper(): y for x, y in zip(amino_acids.AA3C, amino_acids.AA)}


pdb_list = pd.read_csv('data/validation_pdb.csv')['pdb'].unique()

for pdb_id in pdb_list:
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id[3:7], pdir='data/validation_pdb/', file_format='pdb')


class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_model(self, model):
        """Verify if model match the model identifier."""
        # model - only keep model 0
        if model.get_id() == 0:
            return 1
        return 0

    def accept_chain(self, chain):
        if chain.get_id() == self.chain_id:
            return 1
        else:
            return 0

    def accept_residue(self, residue):
        hetatm_flag, resseq, icode = residue.get_id()
        # print(residue.get_id())
        if hetatm_flag != " ":
            # skip HETATMS
            return 0
        if icode != " ":
            warnings.warn(
                "WARNING: Icode %s at position %s" % (icode, resseq), BiopythonWarning
            )
        return 1


for pdb_id_a in pdb_list:
    pdb_id = pdb_id_a[3:7]
    if len(pdb_id_a.split('_')) != 3:
        continue
    pdb_chain = pdb_id_a.split('_')[2]

    data_path = f'data/validation_pdb/pdb{pdb_id.lower()}.ent'
    if not os.path.exists(data_path):
        continue
    p = PDBParser()
    structure = p.get_structure('X', data_path)

    sel = ChainSelect(pdb_chain)
    io = PDBIO()
    io.set_structure(structure)
    io.save(f'data/validation_pdb/chain/{pdb_id}{pdb_chain}.pdb', sel)


