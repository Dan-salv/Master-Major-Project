# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:47:10 2022

@author: Daals
"""


import sys
from sys import argv
sys.path.insert(1, "./") 
import argparse 
import re
import math
from math import sqrt, atan2, degrees
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from os import listdir
from os.path import isfile, join
from module.redo_pck import filesystem, protein_structure, restraints
from tqdm import tqdm 
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio.PDB.vectors import calc_dihedral
import warnings

warnings.simplefilter('ignore', PDBConstructionWarning)


def parse_args():
    #####################
    # START CODING HERE #
    #####################
    # Implement a simple argument parser (WITH help documentation!) that parses
    # the information needed by main() from commandline. 

    parser = argparse.ArgumentParser(prog='python3 filter_rsccs.py',
                                     formatter_class=argparse.RawTextHelpFormatter, description=
                                     '  Read and parse .json files.\n'
                                     '  Example syntax:\n'
                                     '    python3 filter_rsccs.py file_path \n')

    parser.add_argument('-i', dest='input_pdb',
                        help='directory with input data')
    parser.add_argument('-d', dest='input_json_directory', default='./data/density_fit',
                        help='directory with input data')
    parser.add_argument('-o', dest='out_dir', default='./output.txt',
                        help='path to a directory where output files are saved\n'
                             '  (directory will be made if it does not exist)')
   
    # parser.add_argument(?)
    # parser.add_argument(?)
    # parser.add_argument(?)

    args = parser.parse_args()

    return args

def get_jsonpath(pdb_id):
    
    path_dict = {}
    dir_list = ["besttls", "final"]
    
    file = filesystem(pdb_id)

    for model in dir_list:
        
        path_file = file.get_json_path(model)
        
        if not model in path_dict:
            path_dict[model] = {}
            
        path_dict[model] = path_file
        
    return path_dict
    

def get_pdb_file(pdb_id):
    
    pdbid = filesystem(pdb_id)
    
    besttls_path = pdbid.get_besttls_pdb()
    final_path = pdbid.get_final_pdb()
    
    return besttls_path, final_path
    
        
def get_all_df(file):
    data = pd.json_normalize(file, max_level = 0)
    #features under the pdb feature
    pdb = pd.json_normalize(data["pdb"])
    
    #compID = N terminal aminoacid from the peptide bond change : str
    #seqNum = N_Res_Num : int
    #strandID = Chain : str
    pdb = pdb[["strandID","seqNum","compID","insCode"]]
    #seqID = seqID - Not the same to N_Res_Num!!!!
    #RSCCS = Real space correlation coefficient
    features = [pdb, data[["asymID","seqID","RSCCS"]]]
    
    df = pd.concat(features, axis = 1)
    #remove water molecules
    df = df[df["compID"].str.contains("HOH")==False]
    #remove residues with seqID == 0
    df = df[df["seqID"] != 0]
    
    df = df.mask(df == "")
    
    return df

def get_rsccs_dict(file):
    
    rsccs_dict = {}
    aa_dict = {}
    
    for residue in file:
        
        chain = residue["asymID"]
        seqID = residue["seqID"]
        aa = [residue["pdb"]["seqNum"], residue["pdb"]["compID"], residue["pdb"]["insCode"]]
        
        
        #build the dictionary rsccs_dict[chain][seqID] = rsccs
        if not chain in rsccs_dict:
            rsccs_dict[chain] = {}
            aa_dict[chain] = {}
        if not seqID in rsccs_dict[chain]:
            rsccs_dict[chain][seqID] = {}
            aa_dict[chain][seqID] = {}
        
        rsccs_dict[chain][seqID] = residue["RSCCS"]
        aa_dict[chain][seqID] = aa
        
    
    return rsccs_dict, aa_dict

def read_jsonfile(filename):
    
    with open(filename, "r") as f:
        
        file = json.loads(f.read())
        
        #get dataframe from all interested residues
        df = get_all_df(file)
        rsccs_dict, aa_dict = get_rsccs_dict(file)
        
    return df, rsccs_dict, aa_dict

def get_next_rsccs(row, rsccs_dict):
    
    chain = row.asymID
    seqID = row.seqID
    
    try:
       next_rsccs = rsccs_dict[chain][seqID + 1]
       
    except KeyError:
        next_rsccs = np.nan
        
    return next_rsccs

def get_next_aa(row, aa_dict):
    
    chain = row.asymID
    seqID = row.seqID
    
    try:
       next_aa = aa_dict[chain][seqID + 1] 
       
    except KeyError:
        next_aa = [np.nan,np.nan,np.nan]
        
    return next_aa
    
def get_next_rsccs_df(file, pdb_id):

    
    df , rsccs_dict, aa_dict = read_jsonfile(file)
    
    if not len(df.index) == 0:

        df["aa_p1"] =  df.apply(lambda x:
                                get_next_aa(x, aa_dict),axis=1)
            
        df_aa = pd.DataFrame(df['aa_p1'].tolist(), columns=['seqNum_p1', 'compID_p1', 'insCode_p1'])
        
        df.drop("aa_p1",inplace=True, axis=1)
        
        features = [df,df_aa]
        
        df = pd.concat(features , axis = 1)
        
        
        df["RSCCS_p1"] = df.apply(lambda x:
                                get_next_rsccs(x, rsccs_dict),axis=1)
        
        
        #remove nan values in RSCCS_p1 column
        df = df.dropna(subset=["RSCCS_p1",'seqNum_p1'])
        
        df = df.mask(df == "")
        
        df['seqNum_p1'] = df['seqNum_p1'].astype("int") 
        #get sum of RSCCS for all residues of a peptide
        df['RSCCS_sum'] = df[["RSCCS","RSCCS_p1"]].sum(axis=1)
    
    else:
        
        raise ValueError(f'Empty json file for pdb file: {pdb_id}')
    
    return df

def ramdomize_peptides(df):
    
    #filter rows where all the residues from the peptide have a value greater or equal to 0.8
    #df = df[(df['RSCCS_final'] >= 0.8) & (df['RSCCS_p1_final'] >= 0.8)]
    
    df = df.drop(columns = ["RSCCS_besttls","RSCCS_p1_besttls"])
    
    df_random = df.groupby('asymID').apply(lambda x: x.sample(frac = 0.04)).reset_index(drop=True)
    
    return df_random

def filter_rsccs(besttls_df,final_df,pdb_id):
    
    #switch to filter peptides with RSCCS_sum_final bigger or equal than RSCCS_sum_besttls
    
    
    no_filter_rsccs = pd.merge(besttls_df, final_df, how = "inner", 
                             on = ["strandID","seqNum","compID",'seqNum_p1', 'compID_p1','insCode_p1',"insCode","asymID"],
                             suffixes=('_besttls', '_final'))
           
    
    no_filter_df = ramdomize_peptides(no_filter_rsccs)
    
    no_filter_df["pdb_id"] = str(pdb_id)
    
    
    return no_filter_df
    
def print_dataframe(file, df ,out_dir, pdb_id):
    
    if not len(df.index) == 0:
        pdb_dir = os.path.join(out_dir,pdb_id)
                
        peptide_dir = os.path.join(pdb_dir, "peptide_conformation")
        
        if not os.path.isdir(peptide_dir):
            os.makedirs(peptide_dir)
        
        filename= os.path.join(peptide_dir,file)
        
        df.to_csv(filename + ".csv", index = False)

def run_script(pdb_id, out_dir):
    
    json_path_dict = get_jsonpath(pdb_id)
    
    if Path(json_path_dict["besttls"]).exists() and Path(json_path_dict["final"]).exists() :
        print(pdb_id)
        besttls_df = get_next_rsccs_df(json_path_dict["besttls"], pdb_id)
        final_df = get_next_rsccs_df(json_path_dict["final"], pdb_id)                             
        
        no_filter_df = filter_rsccs(besttls_df,final_df,pdb_id)

        
        
        print_dataframe(f"{pdb_id}_random_no_filter", no_filter_df, out_dir, pdb_id)

        
        
def main():
    args = parse_args()
    #f_out = args.out_dir    
    out_dir = args.out_dir

    
    run_script(args.input_pdb, out_dir)
    
    
   
if __name__ == '__main__':
    main()
