# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:20:05 2022

@author: Daals
"""

import sys
sys.path.insert(1, './module')
from sys import argv
import subprocess
import argparse 
import re
import math
from math import sqrt, atan2, degrees
import os
import json
import gzip
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from contextlib import contextmanager
from Bio.PDB.vectors import calc_dihedral, rotaxis2m, Vector, m2rotaxis
from Bio.PDB import parse_pdb_header
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
import inspect
from tqdm import tqdm
import multiprocessing as mp
from redo_pck import filesystem, protein_structure
from pdbecif.mmcif_io import CifFileReader


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

    parser.add_argument('-d', dest='input_directory',
                        help='directory with input data')
    parser.add_argument('-i', dest='input_file_directory',
                        help='directory with input data')
    parser.add_argument('-o', dest='out_dir', default='./output.txt',
                        help='path to a directory where output files are saved\n'
                             '  (directory will be made if it does not exist)')
   
    # parser.add_argument(?)
    # parser.add_argument(?)
    # parser.add_argument(?)

    args = parser.parse_args()

    return args

def get_pdbid_list(input_file):
    path = os.path.join(input_file, "pdb_id")
    pdb_id_list = []
    
    with open(path) as file:
        
        pdb_id_list = file.read().splitlines()
        
    return pdb_id_list

def get_unp_dict (pdb_id_list):
    unp_dict = {}
    
    for pdb_id in tqdm(pdb_id_list):
        
        try:
            cif_header = CifFileReader().read(filesystem(pdb_id).get_final_cif(), ignore=['_atom_site'])
            
            try:
                db_list = cif_header[pdb_id.upper()]["_struct_ref"]["db_name"]
                accesion_number_list = cif_header[pdb_id.upper()]["_struct_ref"]["pdbx_db_accession"]
                
                if not isinstance(db_list, list):
                    db_list = [db_list]
                    accesion_number_list = [accesion_number_list]
                    
                unp_list = set()
                for db, accesion_number in zip(db_list, accesion_number_list):
                    
                    if not db == "UNP":
                        continue
                    
                    else:
                        unp_list.add(accesion_number)
                
                for unp_id in unp_list:
                    
                    if not unp_id in unp_dict:
                        unp_dict[unp_id] = []
                        
                    unp_dict[unp_id].append(pdb_id)
                
                    
                    
            except KeyError:
                continue
        
        except SystemExit:
            continue
    
    return unp_dict


def run_script(input_directory, input_file, out_dir):
    
    pdb_id_list = get_pdbid_list(input_file)
    
    unp_dict = get_unp_dict(pdb_id_list)
    
    json_object = json.dumps(unp_dict, indent=4)
    
    out_file = os.path.join(out_dir, "unp_pdb_map.json")
    
    with open(out_file, "w") as outfile:
        outfile.write(json_object)
    
    
    
    
    
def main():
    args = parse_args()
    #f_out = args.out_dir
    print("")
    print("Analysis Started")  
    print("_______________________")
    print("")
    
    
    run_script(args.input_directory, args.input_file_directory, args.out_dir)
    
    
   
if __name__ == '__main__':
    main()

