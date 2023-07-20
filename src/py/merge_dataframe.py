# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:47:10 2022

@author: Daals
"""

from sys import argv
import argparse 
import os
import json
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm 
from pathlib import Path




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

    parser.add_argument('-d', dest='input_directory',
                        help='directory with input data')
    parser.add_argument('-i', dest='input_file_directory', default='./input.txt',
                        help='directory with input data')
    parser.add_argument('-o', dest='out_dir', default='./output.txt',
                        help='path to a directory where output files are saved\n'
                             '  (directory will be made if it does not exist)')
   
    # parser.add_argument(?)
    # parser.add_argument(?)
    # parser.add_argument(?)

    args = parser.parse_args()

    return args


def run_script(input_directory,filename,out_file):
    
    pdb_id_list = [f for f in listdir(input_directory)]
    
    path_list = [os.path.join(input_directory,pdb_id) for pdb_id in pdb_id_list]
    
    df = pd.DataFrame()
    
    for path, pdb_id in tqdm(zip(path_list, pdb_id_list),  total = len(pdb_id_list)):
        
        #file = f"{pdb_id}_{filename}.csv"
        
        file = f"{filename}.csv"
        
        dir_path = os.path.join(path,"peptide_conformation")
        
        path_file = os.path.join(dir_path,file)
        
        if Path(path_file).exists():
            
            #print(path_file)
        
            df_tmp = pd.read_csv(path_file, dtype = {'pdb_id': str})
            
            df = pd.concat([df, df_tmp],ignore_index=True)
        
    df.to_csv(out_file + ".csv", index = False)
    
    

def main():
    args = parse_args()
    #f_out = args.out_dir
    print("")
    print("Merging dataframes")  
    print("_______________________")
    print("")
    
    out_dir = args.out_dir
    run_script(args.input_directory,args.input_file_directory, out_dir)
    
    
   
if __name__ == '__main__':
    main()
