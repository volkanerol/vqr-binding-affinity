# -*- coding: utf-8 -*-
# Author: Volkan Erol (Marmara University)
# Contact: volkan.erol@gmail.com

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': Descriptors.MolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol),
        'nRotB': Lipinski.NumRotatableBonds(mol),
        'nArom': Lipinski.NumAromaticRings(mol)
    }

def preprocess_dataset(csv_path, output_path):
    df = pd.read_csv(csv_path)
    desc_rows = []
    for smi in df['SMILES']:
        d = compute_descriptors(smi)
        if d:
            desc_rows.append(d)
    desc_df = pd.DataFrame(desc_rows)
    merged = pd.concat([df, desc_df], axis=1)
    merged.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")
