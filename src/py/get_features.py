# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:20:05 2022

@author: Daals
"""

import sys
sys.path.insert(1, "./")  

from sys import argv
import subprocess
import argparse 
import re
import math
from math import sqrt, atan2, degrees
import os
import json
import gzip
import random
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from contextlib import contextmanager
from Bio.PDB.vectors import calc_dihedral, rotaxis2m, Vector, m2rotaxis
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
import inspect
from tqdm import tqdm
import multiprocessing as mp
from module.redo_pck import filesystem, protein_structure, restraints




#print(inspect.getfile(MMCIFParser))
def parse_args():
    #####################
    # START CODING HERE #
    #####################
    # Implement a simple argument parser (WITH help documentation!) that parses
    # the information needed by main() from commandline. 

    parser = argparse.ArgumentParser(prog='python3 get_features.py',
                                     formatter_class=argparse.RawTextHelpFormatter, description=
                                     '  Read and parse .scm files.\n'
                                     '  Example syntax:\n'
                                     '    python3 read_scm.py file_path \n')

    parser.add_argument('-i', dest='input_pdb_id',
                        help='directory with input data')
    parser.add_argument('-d', dest='input_file_directory',
                        help='directory with input data')
    parser.add_argument('-o', dest='out_dir', default='./output.txt',
                        help='path to a directory where output files are saved\n'
                             '  (directory will be made if it does not exist)')
   
    # parser.add_argument(?)
    # parser.add_argument(?)
    # parser.add_argument(?)

    args = parser.parse_args()

    return args


def read_csv(filename):
    
    data = pd.read_csv(filename, dtype={'pdb_id': str})
    
    #list with all pdb_id files where a change was madeF
    
    pdb_id_list = data["pdb_id"].unique()
    
    return data, pdb_id_list

def get_random_df(row, res_dict):
    
    #print(res_dict)
    asymID = row.asymID
    row_tmp = row
    
    #draw random seqID from same asymID
    seqID = random.choice(list(res_dict[asymID]))
    
    compID = res_dict[asymID][seqID][0]
    seqNum = res_dict[asymID][seqID][1]
    insCode = res_dict[asymID][seqID][2]
    
    compID_p1 = res_dict[asymID][seqID + 1][0]
    seqNum_p1 = res_dict[asymID][seqID + 1][1]
    insCode_p1 = res_dict[asymID][seqID + 1][2]
    
    row_tmp.seqID_final = seqID
    
    row_tmp.compID = compID
    row_tmp.seqNum = seqNum
    row_tmp.insCode = insCode
    
    row_tmp.compID_p1 = compID_p1
    row_tmp.seqNum_p1 = seqNum_p1 
    row_tmp.insCode_p1 = insCode_p1
    
    row_tmp.conformation_type = "random"
        

    return row
    
    
def get_residue_id(chain, residue, het_res):
    
    #print(residue)
    inscode = residue[2]
    if isinstance(inscode, float) or inscode == "":
        inscode = " "
        
    try:
        res_name = "H_" + het_res[chain][residue[1]]
        residue_ID = (res_name,residue[1],inscode)
        
    except KeyError:
        residue_ID = (" ",residue[1],inscode)
        
    return residue_ID

def get_pep_res_id(chain, n_seq_id , pos_res, pos_pep, res_dict, het_res):
    
    #print(res_dict)
    pep_seq_id = [n_seq_id, n_seq_id + pos_res]
    
    # AA,Res_Num,inscode from the interested peptide
    pep_res = [res_dict[chain][seq_id + pos_pep] for seq_id in pep_seq_id]

    #correct residue id from the interested peptide
    pep_res_id = [get_residue_id(chain, res, het_res) for res in pep_res]
    
    return pep_res_id[0] , pep_res_id[1]

def get_feature_name(pos_pep, feature):
    
    if pos_pep < 0:
        
        feature_name = feature + "_m" + str(abs(pos_pep))
        
    elif pos_pep == 0:
        feature_name = feature + "_r0"
        
    elif pos_pep > 0:
        feature_name = feature + "_p" + str(abs(pos_pep))
        
    return feature_name

def calc_angle(coord_1,coord_2,coord_3):
    
    coord_1 = coord_1.get_array()
    coord_2 = coord_2.get_array()
    coord_3 = coord_3.get_array()
    # 
    v1 = coord_1 - coord_2
    v2 = coord_3 - coord_2
    
    
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    dot_product = np.dot(v1,v2)
    cross_product = np.cross(v1,v2)
    
    unitary_vector = cross_product / np.linalg.norm(cross_product)
    
    cos = dot_product / (norm_v1 * norm_v2)
    sin = np.dot(cross_product/ (norm_v1 * norm_v2) , unitary_vector)
    
    angle = degrees(atan2(sin, cos))
    
    return angle
 
def get_pseudo_cb_gly(residue):
    
    """Return a pseudo CB vector for a Gly residue (PRIVATE).
        The pseudoCB vector is centered at the origin.
        CB coord=N coord rotated over -120 degrees
        along the CA-C axis.
    """
    
    n_v = residue["N"].get_vector()
    c_v = residue["C"].get_vector()
    ca_v = residue["CA"].get_vector()
    
    # center at origin
    origin_n_v = n_v - ca_v
    origin_c_v = c_v - ca_v
    # rotation around c-ca over -120 deg
    rot = rotaxis2m(-math.pi * 2 / 3 , origin_c_v)
    cb_origin_v = origin_n_v.left_multiply(rot)
    
    norm_cb_origin_v = cb_origin_v / np.linalg.norm(cb_origin_v)
    
    #extension of cb using as reference the ca-cb restraint of alanine
    
    norm_pseudo_cb = norm_cb_origin_v.get_array() * 1.52
    
    
    # move back to ca position
    pseudo_cb = Vector(norm_pseudo_cb) + ca_v
    
    return pseudo_cb


def get_coords(peptide, residues, atoms):
    
    coord_list = []
    
    for residue, atom in zip(residues, atoms):
        
        #if a cb coord is needed for a gly, calculation of a pseudo-gly is needed
        if peptide[residue].get_resname() == "GLY" and atom == "CB":
            
            coord = get_pseudo_cb_gly(peptide[residue])
            
        else:
            
            coord = peptide[residue][atom].get_vector()
        
        coord_list.append(coord)
        
    
    return tuple(coord_list)


def calculate_bond_length(row, atom1, atom2, res_dict, het_res, model, feature):
    # seq ids from the interested peptide 
    
    pdb_id = row.pdb_id
    chain = row.strandID
    asymID = row.asymID
    seq_id = row.seqID_besttls
    raw_atoms = [atom1, atom2]
    
    atoms = [raw_atom.split("_")[0] for raw_atom in raw_atoms]
    terminal = [raw_atom.split("_")[1] for raw_atom in raw_atoms]
    
    for pos_pep in range(-1,3):
        
        feature_name = get_feature_name(pos_pep, feature)
            
        try:
            #get N_ID and C_ID
            N_ID , C_ID = get_pep_res_id(asymID, seq_id , 1, pos_pep, res_dict, het_res)
            
            tmp_dict = {"N": N_ID, "C": C_ID}
            residues = [tmp_dict[residue] for residue in terminal]
            
            #coord_1 = model[chain][residues[0]][atom[0]]
            #coord_2 = model[chain][residues[1]][atom[1]]
            
            coord_1, coord_2 = get_coords(model[chain], residues, atoms)
            
            row[feature_name] = np.linalg.norm(coord_1 - coord_2)
            
        except KeyError:
            
            row[feature_name] = np.nan
        
    return row
    
def calculate_bond_angle(row, atom1, atom2, atom3 ,res_dict, het_res, model, feature):
    
    pdb_id = row.pdb_id
    chain = row.strandID
    asymID = row.asymID
    seq_id = row.seqID_besttls
    raw_atoms = [atom1, atom2, atom3]
    
    atoms = [raw_atom.split("_")[0] for raw_atom in raw_atoms]
    terminal = [raw_atom.split("_")[1] for raw_atom in raw_atoms]
    
    for pos_pep in range(-1,3):
        
        feature_name = get_feature_name(pos_pep, feature)
        
        try:
            #get N_ID and C_ID
            N_ID , C_ID = get_pep_res_id(asymID, seq_id , 1, pos_pep, res_dict, het_res)
            
            tmp_dict = {"N": N_ID, "C": C_ID}
            residues = [tmp_dict[residue] for residue in terminal]
            
            coord_1, coord_2, coord_3 = get_coords(model[chain], residues, atoms)    
            
            row[feature_name] = calc_angle(coord_1,coord_2,coord_3)
            
        
            
        except KeyError:
            
            row[feature_name] = np.nan
           
    return row
           
def calculate_torsion_angle(row, pos_res,atom1, atom2, atom3 ,atom4, res_dict, het_res, model, feature):
    
    pdb_id = row.pdb_id
    chain = row.strandID
    asymID = row.asymID
    seq_id = row.seqID_besttls
    raw_atoms = [atom1, atom2, atom3, atom4]
    
    atoms = [raw_atom.split("_")[0] for raw_atom in raw_atoms]
    terminal = [raw_atom.split("_")[1] for raw_atom in raw_atoms]
    
    for pos_pep in range(-1,3):
        
        feature_name = get_feature_name(pos_pep, feature)
        
        try:
            #get N_ID and C_ID
            N_ID , C_ID = get_pep_res_id(asymID, seq_id , pos_res, pos_pep, res_dict, het_res)
            
            tmp_dict = {"N": N_ID, "C": C_ID}
            residues = [tmp_dict[residue] for residue in terminal]
            #print(residues)
            #print(pdb_id,chain,N_ID , C_ID)
            
            coord_1, coord_2, coord_3, coord_4 = get_coords(model[chain], residues, atoms)    
            
            row[feature_name] = degrees(calc_dihedral(coord_1, coord_2, coord_3, coord_4))
            
        except KeyError:
            
            row[feature_name] = np.nan
           
    return row

def get_bfactors_paramaters(model):
    
    b_factor_model = []
    backbone_atoms = ["CA", "C", "N", "O"]
    
    for residue in model.get_residues():
        
        if residue.has_id("CA") and residue.has_id("C") and residue.has_id("N") and residue.has_id("O"):
            
            b_factor_residue = [residue[backbone_atom].get_bfactor() for backbone_atom in backbone_atoms]
            b_factor_model += b_factor_residue
    
    mean = np.mean(b_factor_model)
    sd = np.std(b_factor_model)
    
    return mean, sd


def calculate_normalized_b_factor(row, pos_res, atom1, res_dict, het_res, model, feature, mean, sd):
    
    pdb_id = row.pdb_id
    chain = row.strandID
    asymID = row.asymID
    seq_id = row.seqID_besttls
    raw_atoms = [atom1]
    
    atoms = [raw_atom.split("_")[0] for raw_atom in raw_atoms]
    terminal = [raw_atom.split("_")[1] for raw_atom in raw_atoms]
    
    for pos_pep in range(-1,3):
        
        feature_name = get_feature_name(pos_pep, feature)
        
        try:
            #get N_ID and C_ID
            N_ID , C_ID = get_pep_res_id(asymID, seq_id , 1, pos_pep, res_dict, het_res)
            
            tmp_dict = {"N": N_ID, "C": C_ID}
            residues = [tmp_dict[residue] for residue in terminal]
            
            #normalized b factor <- b_factor - mean / std
            b_factor = model[chain][residues[0]][atoms[0]].get_bfactor()
            norm_b_factor = (b_factor - mean) / sd
            
            row[feature_name] = norm_b_factor
            
        except KeyError:
            
            row[feature_name] = np.nan
           
    return row

def obtain_secondary_structure(row, dssp_dict,  feature):

    asymID = row.asymID
    seq_id = row.seqID_besttls

    for pos_pep in range(-1,3):
        
        feature_name = get_feature_name(pos_pep, feature)      
        
        try:
            row[feature_name] = dssp_dict[asymID][seq_id + pos_pep]
            
        except KeyError:      
            row[feature_name] = np.nan

    return row

def obtain_hbond_energy(row, hbond_dict):

    asymID = row.asymID
    seq_id = row.seqID_besttls
    
    try:
        row["donor_1"] = hbond_dict[asymID][seq_id]["donor_1_energy"]
        row["donor_2"] = hbond_dict[asymID][seq_id]["donor_2_energy"]

    except KeyError: 
        row["donor_1"] = np.nan
        row["donor_2"] = np.nan
    
    try:
        row["acceptor_1"] = hbond_dict[asymID][seq_id + 1]["acceptor_1_energy"]
        row["acceptor_2"] = hbond_dict[asymID][seq_id + 1]["acceptor_2_energy"]

    except KeyError:
        row["acceptor_1"] = np.nan
        row["acceptor_2"] = np.nan
        
    return row

def obtain_ramachandran_zscores(row, ramachandran_dict,feature):

    asymID = row.asymID
    seq_id = row.seqID_besttls

    for pos_pep in range(-1,3):

        feature_name = get_feature_name(pos_pep, feature)

        try:
            row[feature_name] = ramachandran_dict[asymID][seq_id + pos_pep]['ramachandran']
            
        except KeyError:      
            row[feature_name] = np.nan
    
    return row

def get_ideal_values(dict_restraints, res, res_p1, atom, feature):
    
    #remove r0
    
    if "r0" in atom:
        atom = atom.split("_")[0]
        
    try:
        
        ideal_value = dict_restraints[res][feature][atom]["value"]

    except KeyError:
        #if residue p1 of a peptide is a  proline or a hidroxyproline value is in PTRANS 
        if not "_" in atom:
            if res_p1 == "PRO" or res_p1 == "HYP":
                
                ideal_value = dict_restraints['PEPTIDE']["PTRANS"][feature][atom]["value"]
                
            else:
                
                ideal_value = dict_restraints['PEPTIDE']["TRANS"][feature][atom]["value"]       
                
        else:
            atom = atom.split("_")[0]

            
            ideal_value = dict_restraints[res_p1][feature][atom]["value"]
    
    ideal_value = float(ideal_value)
    
    return ideal_value

def calculate_wh(row, dict_restraints, quotient, contributions, feature):
    
    res = row.compID
    res_p1 = row.compID_p1
    
    ideal_values = [get_ideal_values(dict_restraints, res, res_p1, atom, feature) for atom in contributions]
    
    wh_sum = sum([(row[atom] - ideal_value) * quotient[atom] for atom, ideal_value in zip(contributions, ideal_values)])
    
    return wh_sum

def calculate_d_phi(row):
    
    phi_angle = row.phi_p1
    
    if phi_angle < 0:
        d_phi = 0
        
    else:
        d_phi = np.sin(math.radians(phi_angle))
        
    return d_phi


def get_wh(df):
    
    dict_restraints = restraints().get_restraints_dict()

    dict_parameters = restraints().get_parameters_wh_dict()
    
    #list of bonds needed for wh algorithm
    bond = ["nca_r0", "cac_r0", "co_r0", "cn_r0", "nca_p1", "cac_p1"]
    #list of angles needed for wh algorithm
    angle =  ["ncac_r0","caco_r0","cacn_r0","cnca_r0","ocn_r0","ncac_p1"]
    
    df["dbond"] = df.apply(lambda x:
             calculate_wh(x, dict_restraints, pd.Series(dict_parameters["bond"]) ,bond, "bond"), axis = 1)
    
    df["dangle"] = df.apply(lambda x:
             calculate_wh(x, dict_restraints, pd.Series(dict_parameters["angle"]) ,angle, "angle"), axis = 1)
        
    df["dphi"] = df.apply(lambda x:
             calculate_d_phi(x), axis = 1)
    
    df["dcaca"] = df.apply(lambda x: (x.caca_r0 - 3.804) * dict_parameters["caca"]["caca_r0"], axis = 1)
        
    
 
    return df

           
def get_features(df, res_dict, het_res, dssp_dict,hbond_dict,ramachandran_dict,model):
    
    """ peptide bond distance between Carbon from carbonil residue i and Nitrogen from residue i+1 """

    df = df.apply(lambda x:
             calculate_bond_length(x ,"C_N","N_C",res_dict, het_res, model,"cn"),axis=1)

    """ peptide bond distance between Nitrogen of residue i and Carbon alfa from residue i """

    df = df.apply(lambda x:
              calculate_bond_length(x ,"N_N","CA_N",res_dict, het_res, model,"nca"),axis=1)
        
    """ peptide bond distance between Carbon alfa of residue i and carbonil carbon from residue i """

    df = df.apply(lambda x:
              calculate_bond_length(x ,"CA_N","C_N",res_dict, het_res, model,"cac"),axis=1)
        
    """ peptide bond distance between carbonil carbon of residue i and oxygen from residue i """

    df = df.apply(lambda x:
              calculate_bond_length(x,"C_N","O_N",res_dict, het_res, model,"co"),axis=1)
        
    """ backbone distance between carbon alfa of residue i and carbon alfa from residue i + 1 """  

    df = df.apply(lambda x:
             calculate_bond_length(x,"CA_N","CA_C",res_dict, het_res, model,"caca"),axis=1)
        
    """ backbone distance between oxygen of residue i and oxygen from residue i + 1 """    
    
    df = df.apply(lambda x:
             calculate_bond_length(x,"O_N","O_C",res_dict, het_res, model,"o_o"),axis=1)
        
    """ backbone distance between carbon beta of residue i and carbon beta from residue i + 1 """      
    
    df = df.apply(lambda x:
             calculate_bond_length(x,"CB_N","CB_C",res_dict, het_res, model,"cbcb"),axis=1)
    
    """ peptide bond angle between nitrogen, alfa carbon and carbonil carbon of residue i """
    
    df = df.apply(lambda x:
              calculate_bond_angle(x,"N_N","CA_N","C_N" ,res_dict, het_res, model,"ncac"),axis=1) 
        
    """ peptide bond angle between alfa carbon, carbonil carbon and carbonil oxygen of residue i """
    
    df = df.apply(lambda x:
              calculate_bond_angle(x,"CA_N","C_N","O_N" ,res_dict, het_res, model,"caco"),axis=1) 
        
    """ peptide bond angle between carbonil oxygen, carbonil carbon of residue i and nitrogen from residue i+1 """
    
    df = df.apply(lambda x:
              calculate_bond_angle(x,"O_N","C_N","N_C" ,res_dict, het_res, model,"ocn"),axis=1) 
    
    """ peptide bond angle between alfa carbon, carbonil carbon of residue i and nitrogen from residue i+1 """
    
    df = df.apply(lambda x:
              calculate_bond_angle(x,"CA_N","C_N","N_C" ,res_dict, het_res, model,"cacn"),axis=1) 
    
    """ peptide bond angle between carbonil carbon of residue i and nitrogen, alfa carbon from residue i+1 """
    
    df = df.apply(lambda x:
              calculate_bond_angle(x,"C_N","N_C","CA_C" ,res_dict, het_res, model,"cnca"),axis=1)
        
    """ peptide bond angle between nitrogen, alfa carbon, and beta carbon of residue i"""
    
    df = df.apply(lambda x:
              calculate_bond_angle(x,"N_N","CA_N","CB_N" ,res_dict, het_res, model,"ncacb"),axis=1) 
    
    """ peptide bond angle between nitrogen, alfa carbon, and beta carbon of residue i"""
     
    df = df.apply(lambda x:
              calculate_bond_angle(x,"C_N","CA_N","CB_N" ,res_dict, het_res, model,"ccacb"),axis=1)     
    
    """phi torsion angle"""
    
    df = df.apply(lambda x:
              calculate_torsion_angle(x, -1 ,"C_C","N_N","CA_N","C_N" ,res_dict, het_res, model,"phi"),axis=1) 
    
    """psi torsion angle"""
    
    df = df.apply(lambda x:
              calculate_torsion_angle(x, 1 ,"N_N","CA_N","C_N","N_C" ,res_dict, het_res, model,"psi"),axis=1)
    
    """omega torsion angle """
    
    df = df.apply(lambda x:
              calculate_torsion_angle(x, 1 ,"CA_N","C_N","N_C","CA_C" ,res_dict, het_res, model,"omega"),axis=1)
    
    """occac torsion angle"""

    df = df.apply(lambda x:
              calculate_torsion_angle(x, 1 ,"O_N","C_N","CA_C","C_C" ,res_dict, het_res, model,"occac"),axis=1)
    
    """ torsion angle between oxygen and carbon from carbonyl group of residue i and residue i + 1  """
    
    df = df.apply(lambda x:
              calculate_torsion_angle(x, 1 ,"O_N","C_N","C_C","O_C" ,res_dict, het_res, model,"coco"),axis=1)
    
    """improper ca dihedral angle """
    
    df = df.apply(lambda x:
              calculate_torsion_angle(x, 1 ,"CA_N","N_N","C_N","CB_N" ,res_dict, het_res, model,"imp"),axis=1)
    
    mean, sd = get_bfactors_paramaters(model)
    
    """ normalized n b_factor ratio """
    
    df = df.apply(lambda x:
              calculate_normalized_b_factor(x, 0, "N_N", res_dict, het_res, model, "bn", mean, sd), axis=1)
    
    """ normalized ca b_factor ratio """
    
    df = df.apply(lambda x:
              calculate_normalized_b_factor(x, 0, "CA_N", res_dict, het_res, model, "bca", mean, sd), axis=1)
    
    """ normalized c b_factor ratio """
    
    df = df.apply(lambda x:
              calculate_normalized_b_factor(x, 0, "C_N", res_dict, het_res, model, "bc", mean, sd), axis=1)
    
    """ normalized o b_factor ratio """
    
    df = df.apply(lambda x:
              calculate_normalized_b_factor(x, 0, "O_N", res_dict, het_res, model, "bo", mean, sd), axis=1)
    
    """ hbond """
    df = df.apply(lambda x:
              obtain_hbond_energy(x, hbond_dict),axis=1)
    
    """ ramachandran z-scores """

    df = df.apply(lambda x:
              obtain_ramachandran_zscores(x, ramachandran_dict, 'zram'),axis=1)
    
    """ dssp information """

    df = df.apply(lambda x:
              obtain_secondary_structure(x, dssp_dict, "dssp"),axis=1)
    
    
    df = get_wh(df)

    
    return df

def get_dataframes(data, pdb_id):
    
    df = data.loc[data['pdb_id'] == pdb_id]
    
    if not len(df.index) == 0: 
        
        #cambia el tipo de archivo a cif
        file_pdb = protein_structure(pdb_id, "besttls", "cif")

        #pdb structure with coordinates x,y,z
        structure = file_pdb.get_structure()
        model = structure[0]
        #dictionary with hetero_residues
        het_res = file_pdb.get_het_res()
        #dictionary with dict[chain][seqID] = [compID, seqNum, inscode]
        res_dict = file_pdb.get_res_dict("besttls")
        #dictionary with dict[asymID][seqID] = "secondary_structure"

        ramachandran_dict = file_pdb.get_ramachandran_dict()

        
        try:
            dssp_dict = file_pdb.get_secondary_dict()

        except KeyError:
            dssp_dict = {}

            print(f"dssp dictionary from Pdb file: {pdb_id} is empty ")

        try:
            hbond_dict = file_pdb.get_hbond_dict()
        
        except KeyError:
            hbond_dict = {}
            print(f"hbond dictionary from Pdb file: {pdb_id} is empty ")
            
        df = get_features(df, res_dict, het_res, dssp_dict, hbond_dict, ramachandran_dict, model)
        
        #random_df = get_features(random_df, res_dict, het_res, model)
            
    #return df , random_df
    return df



def print_dataframe(file,df,out_dir):
    
    filename= os.path.join(out_dir,file)
    
    df.to_csv(filename + ".csv", index = False)

def run_script(input_pdb,input_file_directory,out_dir):
    #print(os.getcwd())
    #print(input_file_directory)
    data, pdb_id_list = read_csv(join(input_file_directory,"no_feature_training_set.csv"))

   

    df = pd.DataFrame()
    
    try:
        #df_tmp, random_tmp = get_dataframes(data, pdb_id)
        df = get_dataframes(data, input_pdb)
        #print(df.columns)
        #print(df[["dssp_m1","dssp_r0","dssp_p1","dssp_p2"]])
        #print(df[["conformation_type",'donor_1', "donor_2", 'acceptor_1', 'acceptor_2']])
        #print(df[['occac_m1','occac_r0','occac_p1','occac_p2','zram_m1','zram_r0','zram_p1','zram_p2']])
        
        #random_df = pd.concat([random_df, random_tmp],ignore_index=True)
    except SystemExit:
        print(f"Pdb file: {input_pdb} failed")

    if not len(df.index) == 0:

        out_file = f"training_peptide_data"    
        
        print_dataframe(out_file,df, out_dir)

        print(f"Features from Pdb file: {input_pdb} obtained succesfully ")
    

def main():
    args = parse_args()
    #f_out = args.out_dir
    print("")
    print("Analysis Started")  
    print("_______________________")
    print("")
    
    out_dir = args.out_dir
    
    
    run_script(args.input_pdb_id, args.input_file_directory, out_dir)
    
    
   
if __name__ == '__main__':
    main()