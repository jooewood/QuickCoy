#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:15:44 2022

@author: zdx
"""
from quickcoy_main import smiles_tokenizer
from plot import xDistribution, DfNonZero, df_feature_comparison
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from glob import glob
from utils import get_can_smiles, add_can_smiles, add_decoy_properties,\
    concat_dfs, adjust_order


def draw_smiles_hist(df, title, col='SMILES'): # col: SMILES or can_smiles
    df = add_can_smiles(df)
    smiles = df[col]    
    lens = []
    for s in smiles:
        lens.append(len(smiles_tokenizer(s)))
    xDistribution(lens, title=title)

    df_new = df[['ID', 'SMILES']].copy()
    df_new['SMILES'] = df['can_smiles']
    return df_new

# dataset = 'LIT-PCBA'

# active_files = glob(f'/home/zdx/data/VS_dataset/{dataset}/original_actives/*.csv')

# dfs = []
# for file in tqdm(active_files):
#     df = pd.read_csv(file)
#     dfs.append(df)
# df = pd.concat(dfs)

# names = ['REAL', 'REAL_can', 'REAL_subset_20_deepcoy_properties', 'REAL_subset_20_deepcoy_properties_can', 'DUD-E', 'LIT-PCBA']

# des = []

# for name in names:
#     df = pd.read_csv(f'/home/zdx/data/REAL/{name}.csv')
    
#     smiles = df.SMILES
#     lens = []
#     for s in tqdm(smiles):
#         lens.append(len(smiles_tokenizer(s)))
#     s = pd.Series(lens)
#     des.append(s.describe().astype(int))

# des_df = pd.DataFrame(des)
# des_df['Data'] = names

# xDistribution(lens, title=f'{dataset} active canonical SMILES')

## Analyze DUD-E actives
# dataset = 'DUD-E'
# files = glob(f'/home/zdx/data/VS_dataset/{dataset}/original_actives/*.csv')

# df = concat_dfs(files) # 22805
# # Before Add H 
# df.drop_duplicates(subset='ID', inplace=True)
# res = DfNonZero(df, drop=['ID', 'SMILES', 'label'], figsize1=(12,12), figsize2=(40,10))

# # After Add H
# df = df[['ID', 'SMILES', 'label']]
# df_pro = add_decoy_properties(df)

# res_addH = DfNonZero(df, only=True, drop=['ID', 'SMILES', 'label'], figsize1=(12,12), figsize2=(40,10))

# old_ID = set(df.ID)
# new_ID = set(df_pro.ID)
# miss = old_ID - new_ID

# df.query(f'ID=="{list(miss)[0]}"')

# for file in files:
#     df = pd.read_csv(file)
#     if list(miss)[0] in list(df.ID):
#         print(file)
#         df = df[df['ID']!=list(miss)[0]]
#         df.to_csv(file, index=False)

# # 1st training set vs DUD-E active
# real = pd.read_csv('/home/zdx/data/REAL/REAL_deepcoy_properties.csv')
# dataset = 'DUD-E'
# files = glob(f'/home/zdx/data/VS_dataset/{dataset}/original_actives/*.csv')
# dude_active = concat_dfs(files)

pro=['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
      'nitrogen_n', 'oxygen_n', 'fluorine_n', # 3
      'phosphorus_n', 'sulfur_n', 'chlorine_n', # 3
      'bromine_n', 'iodine_n', 'logP', 'HBA', # 4
      'HBD', 'rings', 'stereo_centers', 'MW', # 4
      'aromatic_rings', 'NRB', 'pos_charge', # 3
      'neg_charge', 'formal_net_charge', 'TPSA', # 3
      'SA', 'QED' # 2
    ]

# df_feature_comparison(dude_active, real, columns=pro, left_name="DUD-E active", right_name="REAL", figsize=(40, 8),
#                       output_file='/home/zdx/project/QuickCoy/Pictures/DUD-E_active_VS_REAL.png')

# 1st quickcoy gererated molecule analysis
df = pd.read_csv('/home/zdx/data/VS_dataset/DUD-E/QuickCoy_filtered_decoy_number.csv')
df_match = df.query('decoy_count==50')
df_mismatch = df.query('decoy_count<50') # 5.5%
df_feature_comparison(df_mismatch, df_match, columns=pro, left_name="mismatch", right_name="match", figsize=(40, 8),
                      output_file='/home/zdx/project/QuickCoy/Pictures/QuickCoy_mismatch_VS_match.png')

df_mismatch = df.query('decoy_count<10') # 5.5%
df_feature_comparison(df_mismatch, df_match, columns=pro, left_name="mismatch", right_name="match", figsize=(40, 8),
                      output_file='/home/zdx/project/QuickCoy/Pictures/QuickCoy_mismatch10_VS_match.png')

