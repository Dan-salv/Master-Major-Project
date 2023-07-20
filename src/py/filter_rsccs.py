# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:47:10 2022

@author: Daals
"""


import sys
sys.path.insert(1, "./")  
from sys import argv
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

def filter_rsccs(besttls_df,final_df):
    
    #switch to filter peptides with RSCCS_sum_final bigger or equal than RSCCS_sum_besttls
    
    switch = False
    
    if not switch:
        filter_rsccs = pd.merge(besttls_df, final_df, how = "inner", 
                             on = ["strandID","seqNum","compID",'seqNum_p1', 'compID_p1','insCode_p1',"insCode","asymID"],
                             suffixes=('_besttls', '_final'))
        
    else:
        #merge based ob same strandID, seqNum, compID, inscode, asymID
        rsccs_sum = pd.merge(besttls_df, final_df, how = "inner", 
                             on = ["strandID","seqNum","compID",'seqNum_p1', 'compID_p1','insCode_p1',"insCode","asymID"],
                             suffixes=('_besttls', '_final'))
    
        #filter rows where RSCCS_sum_final is bigger or equal than RSCCS_sum_besttls
        
        filter_rsccs = rsccs_sum[rsccs_sum["RSCCS_sum_final"] >= rsccs_sum["RSCCS_sum_besttls"]]
        
    #filter rows where all the residues from the peptide have a value greater or equal to 0.8
    #filter_rsccs = filter_rsccs[(filter_rsccs['RSCCS_final'] >= 0.8) & (filter_rsccs['RSCCS_p1_final'] >= 0.8)]
    
    return filter_rsccs
    
def get_cif_model(pdb_id, model):
    
    file_cif = protein_structure(pdb_id, model, "cif")
    
    structure = file_cif.get_structure()

    model = structure[0]
        #dictionary with hetero_residues
    het_res = file_cif.get_het_res()
    
    return model, het_res


def get_residue_id(chain, residue, het_res):
    
    inscode = residue[2]
    
    if isinstance(inscode, float):
        inscode = " "
        
    try:
        res_name = "H_" + het_res[chain][residue[1]]
        residue_ID = (res_name,residue[1],inscode)
        
    except KeyError:
        residue_ID = (" ",residue[1],inscode)
        
    return residue_ID

def calc_omega_angle(n_terminal_residue, c_terminal_residue):
    
    CA_N  =  n_terminal_residue["CA"].get_vector()
    #Carbonil Carbon from N_terminal
    C_N   =  n_terminal_residue["C"].get_vector()
    #Nitrogen from C_terminal
    N_C   =  c_terminal_residue["N"] .get_vector()
    #C alfa from C_terminal
    CA_C  =  c_terminal_residue["CA"].get_vector()
    
    omega_angle = degrees(calc_dihedral(CA_N, C_N, N_C, CA_C))
    
    return omega_angle

def calc_occac_angle(n_terminal_residue, c_terminal_residue):
    
    O_N  =  n_terminal_residue["O"].get_vector() 
    #Carbonil Carbon from N_terminal
    C_N   =  n_terminal_residue["C"].get_vector() 
    #Nitrogen from C_terminal
    CA_C   =  c_terminal_residue["CA"] .get_vector() 
    #C alfa from C_terminal
    C_C  =  c_terminal_residue["C"].get_vector() 
    
    occac_angle = degrees(calc_dihedral(O_N, C_N, CA_C, C_C))
    
    return occac_angle
    

def calculate_torsion_angle(row,model,het_res,feature):
    
    chain = row.strandID
    n_res = [row.compID,row.seqNum,row.insCode]
    c_res = [row.compID_p1,row.seqNum_p1,row.insCode_p1]
    
    N_ID = get_residue_id(chain, n_res, het_res)
    C_ID = get_residue_id(chain, c_res, het_res)
    
    feature_omega = "omega_" + feature
    feature_occac = "occac_" + feature
    
    try:
        n_terminal_residue = model[chain][N_ID]
        c_terminal_residue = model[chain][C_ID]
        
        #discard disordered residues
        if n_terminal_residue.is_disordered() or c_terminal_residue.is_disordered():
            
            row[feature_omega] = np.nan
            row[feature_occac] = np.nan
            
        else:
            omega_angle = calc_omega_angle(n_terminal_residue, c_terminal_residue)
            occac_angle = calc_occac_angle(n_terminal_residue, c_terminal_residue)
            
            if omega_angle < 0:
                omega_angle = omega_angle + 360
                
            if occac_angle < 0:
                occac_angle = occac_angle + 360
            
            row[feature_omega] = omega_angle
            row[feature_occac] = occac_angle
                
    
    except KeyError:
        row[feature_omega] = np.nan
        row[feature_occac] = np.nan
        
    return row    


def get_torsion_angle_df(pdb_id, df):

    besttls_model, besttls_het_res = get_cif_model(pdb_id, "besttls")
    
    final_model, final_het_res = get_cif_model(pdb_id, "final")
    
    df = df.apply(lambda x:
                  calculate_torsion_angle(x, besttls_model, besttls_het_res,"besttls"),axis=1)
        
    df = df.apply(lambda x:
                  calculate_torsion_angle(x, final_model, final_het_res,"final"),axis=1)
    
    if not len(df.index) == 0:
        
        df.dropna(subset=["omega_besttls","occac_besttls","omega_final","occac_final"], inplace=True)
        
        df["delta_omega"] = abs(df["omega_besttls"] - df["omega_final"])
        
        #df["delta_omega"] = df["delta_omega"].apply(lambda x: x + 360 if x < 0 else x)
        
        df["delta_occac"] = abs(df["occac_besttls"] - df["occac_final"])
        
        #df["delta_occac"] = df["delta_occac"].apply(lambda x: x + 360 if x < 0 else x)
    
    return df

def decide_conformation(angle, cut_off):
    
    no_change = [0 + cut_off, 360 - cut_off]
    change = [180 - cut_off, 180 + cut_off]
    
    if angle <= no_change[0] or angle >= no_change[1]:
        return False
    
    elif change[0] <= angle <= change[1]:
        return True
    
    else:
        return np.nan

def decide_isomer_type(isomerism,omega_final):
    
    if not isomerism:
        return "no_isomer"
    
    else:
        if 120 <= omega_final <= 240:
            return "cis_trans"
        
        else:
            return "trans_cis"
        
def decide_conformation_type(isomerism,pepflip):
    
    if not isomerism:
        if not pepflip:  
            return "no_change"
        
        else:
            return "pep_flip"
    
    else:
        if not pepflip:
            return "n_flip"
        
        else:
            return "o_flip"
        
def get_peptide_conformation(df, pdb_id):
    
    if not len(df.index) == 0:
    #check if isomer conformation was made
        df["isomerism"] = df["delta_omega"].apply(lambda x: decide_conformation(x, 60))
        
        #check if peptide flip was made
        df["pepflip"] = df["delta_occac"].apply(lambda x: decide_conformation(x, 75))
        
        #drop irregular peptides
        df.dropna(subset=["isomerism","pepflip"], inplace=True)
    
        
        df = df.astype({"isomerism": 'bool', "pepflip": 'bool'})
        
        #decide isomer type (trans_cis, cis_trans, pep_flip)
        df["isomer_type"] = df.apply(lambda x: decide_isomer_type(x.isomerism, x.omega_final),axis=1)
        
        df["conformation_type"] = df.apply(lambda x: decide_conformation_type(x.isomerism, x.pepflip),axis=1)
        
        df["pdb_id"] = str(pdb_id)
        
        df = df[~df["conformation_type"].str.contains("no_change")]
    
    return df

def print_dataframe(file,df,out_dir):
    
    filename= os.path.join(out_dir,file)
    
    df.to_csv(filename + ".csv", index = False)

def run_script(pdb_id, out_dir):
    
    json_path_dict = get_jsonpath(pdb_id)

    
    if Path(json_path_dict["besttls"]).exists() and Path(json_path_dict["final"]).exists() :

        if os.stat(json_path_dict["besttls"]).st_size == 0 or os.stat(json_path_dict["final"]).st_size == 0 :
            
            print(f"Besttls or final json file empty for pdb file: {pdb_id}")
        
        else:
            
            besttls_df = get_next_rsccs_df(json_path_dict["besttls"], pdb_id)
            final_df = get_next_rsccs_df(json_path_dict["final"], pdb_id)
                            
            filter_rsccs_df = filter_rsccs(besttls_df,final_df)

            torsion_angle_df = get_torsion_angle_df(pdb_id, filter_rsccs_df)

            df = get_peptide_conformation(torsion_angle_df, pdb_id)
            
            if not len(df.index) == 0:
                
                file = f"{pdb_id}_peptide_conformation_no_filter"
                
                pdb_dir = os.path.join(out_dir,pdb_id)
                
                peptide_dir = os.path.join(pdb_dir, "peptide_conformation")
                
                
                if not os.path.isdir(peptide_dir):
                    os.makedirs(peptide_dir)
                
                print_dataframe(file, df, peptide_dir)
                print(f"peptide_conformation file recorded for pdb file: {pdb_id}")
        
    else:
        print(f"The pdb_id {pdb_id} does not contain both besttls and final json files ") 

        

   

def main():
    args = parse_args()
    #f_out = args.out_dir    
    out_dir = args.out_dir
    
    run_script(args.input_pdb, out_dir)
    
    
   
if __name__ == '__main__':
    main()
