# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:49:05 2022

@author: Daals
"""
import os
import sys
from pathlib import Path
from pdbecif.mmcif_io import CifFileReader
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser , MMCIFParser
from Bio.PDB import PDBIO
from Bio.SeqUtils import IUPACData
import numpy as np
import json
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

class filesystem:
    
    def __init__(self, pdbid):
       self.pdbid = pdbid
       
    def get_path(self, filename):
        
        pdbid = self.pdbid
        middle = pdbid[1:3]
        
        pth = f"/DATA/pdb_redo/{middle}/{pdbid}/" 
        f = os.path.join(pth,filename)
        
        if Path(f).exists():
            return f
        else:
           sys.exit(f"{pdbid}: {filename} does not exist")
    
    def get_density_fit_path(self, filename):
        
        pdbid = self.pdbid

        pth = f"/DATA/raw_data/density_fit/{pdbid}/" 
        #pth = f"data/density_fit/{pdbid}/" 
        
        f = os.path.join(pth,filename)
        
        if Path(f).exists():
            return f
        else:
           sys.exit(f"{pdbid}: {filename} does not exist")
    
    def get_alphfold_path(self, filename):
        
        uniprot_id = self.pdbid
        
        middle = uniprot_id[0:2]
        pth = f"/DATA/raw_data/dssp_alphafold/{middle}/" 
        #pth = f"data/density_fit/{pdbid}/" 
        
        f = os.path.join(pth,filename)
        
        if Path(f).exists():
            return f
        else:
           sys.exit(f"{uniprot_id}: {filename} does not exist")
    
    def get_final_pdb(self):
        
        pdbid = self.pdbid
        pdb = self.get_path(f"{pdbid}_final.pdb")
        
        return pdb
    def get_besttls_pdb(self):
        
        pdbid = self.pdbid
        pdb = self.get_path(f"{pdbid}_besttls.pdb.gz")
        
        return pdb
    
    def get_final_cif(self):
        
        pdbid = self.pdbid
        cif = self.get_path(f"{pdbid}_final.cif")
        
        return cif
    
    def get_besttls_cif(self):
        
        pdbid = self.pdbid
        cif = self.get_density_fit_path(f"{pdbid}_besttls.cif")
        
        return cif
    
    def get_besttls_stripped_cif(self):
        
        pdbid = self.pdbid
        cif = self.get_density_fit_path(f"{pdbid}_besttls_stripped.cif")
        
        return cif
    
    def get_dssp_path(self, model):

        pdbid = self.pdbid
        filename =  f"{pdbid}_{model}_dssp.cif"
        dssp_file = self.get_density_fit_path(filename)

        return dssp_file
    
    def get_dssp_alphafold_path(self):

        pdbid = self.pdbid
        filename =  f"AF_{pdbid}_dssp.cif"
        dssp_file = self.get_alphfold_path(filename)

        return dssp_file

    def get_json_path(self, model):
        
        pdbid = self.pdbid
        
        filename =  f"{pdbid}_{model}.json"
        json = self.get_density_fit_path(filename)

        return json
    
    def get_json_tortoize_path(self, model):
        
        pdbid = self.pdbid
        
        filename =  f"{pdbid}_{model}_tortoize.json"
        json = self.get_density_fit_path(filename)

        return json
    

    def read_json_file(self, model):
        
        json_file = self.get_json_path(model)
        
        with open(json_file, "r") as f:
            
            file = json.loads(f.read())
            
        return file
    
    def read_json_tortoize_file(self, model):
        
        json_file = self.get_json_tortoize_path(model)
        
        with open(json_file, "r") as f:
            
            file = json.loads(f.read())
            
        return file
    
    def read_dssp_file(self, model):

        if model == "alphafold":

            dssp_path = self.get_dssp_alphafold_path()
        else:

            dssp_path = self.get_dssp_path(model)
        
        dssp_file = CifFileReader().read(dssp_path)

        return dssp_file

class protein_structure:
    
    def __init__(self, pdbid, model, format):
       
       self.pdbid = pdbid
       self.model = model
       self.format = format
       
    def get_structure(self):
        
        """Parse pdb file"""
        file = filesystem(self.pdbid)
        model = self.model
        format = self.format
        
        if format == "pdb":
            parser = PDBParser()   

            if model == "final":    
                path = file.get_final_pdb()

            elif model == "besttls":
                path = file.get_besttls_pdb()
        
        elif format == "cif":
            parser = MMCIFParser() 

            if model == "final":    
                path = file.get_final_cif()
                    
            elif model == "besttls":
                path = file.get_besttls_cif()

        else:
            sys.exit(f"non existant model")
            
    
        structure = parser.get_structure(self.pdbid, path)
        
        return structure
    
    def get_het_res(self):
        
        """Get dictionary of hetero residues """
        model = protein_structure(self.pdbid, self.model, self.format).get_structure()[0]
    
        # residue = [row["N_AA"], row["N_Res_Num"], row["N_insCode"]]
        het_res = {}

        for residue in model.get_residues():
            
            residue_id = residue.get_id()
            #print(residue.get_full_id())
            #print(residue.get_full_id)
            hetfield = residue_id[0]
            
            if hetfield[0] == "H":
                chain = residue.get_full_id()[2]
                AA = residue.get_resname()
                Res_Num = residue.get_id()[1]
                
                if not chain in het_res:
                    het_res[chain] = {}
                if not Res_Num in het_res[chain]:
                    het_res[chain][Res_Num] = {}
                    
                het_res[chain][Res_Num] = AA
    
        return het_res
    
    def get_res_dict(self, model):
        
        """Get dictionary of residues with structure dict[chain][seqID] = [compID , seqNum, , inscode]"""
        
        path = filesystem(self.pdbid)
        
        file = path.read_json_file(model)
        
        aa_dict = {}
        
        for residue in file:
            
            chain = residue["asymID"]
            seqID = residue["seqID"]
            aa = [residue["pdb"]["compID"],residue["pdb"]["seqNum"], residue["pdb"]["insCode"]]
            
            #build the dictionary rsccs_dict[chain][seqID] = rsccs
            if not chain in aa_dict:
                aa_dict[chain] = {}
            if not seqID in aa_dict[chain]:
                aa_dict[chain][seqID] = {}
            
            aa_dict[chain][seqID] = aa
            
        
        return  aa_dict
    
    def get_ramachandran_dict(self):

        path = filesystem(self.pdbid)
        
        file = path.read_json_tortoize_file(self.model)

        residues = file["model"]["1"]['residues']
        ramachandran_dict = {}

        #print(residues[0].keys())

        for residue in residues:

            asymID = residue["asymID"]
            seqID = residue["seqID"]

            if not asymID in ramachandran_dict:
                ramachandran_dict[asymID] = {}
            if not seqID in ramachandran_dict[asymID]:
                ramachandran_dict[asymID][seqID] = {}
            
            ramachandran_dict[asymID][seqID]['ramachandran'] = residue['ramachandran']['z-score']
            
            try:
                ramachandran_dict[asymID][seqID]['torsion'] = residue['torsion']['z-score']

            except KeyError:
                ramachandran_dict[asymID][seqID]['torsion'] = np.nan

            
        return ramachandran_dict
    
    def get_secondary_dict(self):

        ss_dict = {}

        pdb_id = self.pdbid
        path = filesystem(pdb_id)
        model = self.model

        dssp = path.read_dssp_file(model)
            
        key_id = list(dssp.keys())[0]

        asym_id_list = dssp[key_id]["_dssp_struct_summary"]['label_asym_id']
        seq_id_list = [int(seq_id) for seq_id in dssp[key_id]["_dssp_struct_summary"]['label_seq_id']]
        ss_list = [ss.replace(".","loop") for ss in dssp[key_id]["_dssp_struct_summary"]['secondary_structure']]

        for asym_id, seq_id, ss in zip(asym_id_list,seq_id_list,ss_list):
            
            if not asym_id in ss_dict:
                ss_dict[asym_id] = {}
            if not seq_id in ss_dict[asym_id]:
                ss_dict[asym_id][seq_id] = {}

            ss_dict[asym_id][seq_id] = ss
        
        

        return ss_dict
    
    def get_hbond_dict(self):

        hbond_dict = {}

        pdb_id = self.pdbid
        path = filesystem(pdb_id)
        model = self.model

        dssp = path.read_dssp_file(model)
            
        key_id = list(dssp.keys())[0]
        hbond = dssp[key_id]['_dssp_struct_bridge_pairs']

        asym_id_list = hbond['label_asym_id']
        seq_id_list = [int(seq_id) for seq_id in hbond['label_seq_id']]
        acceptor1_list = np.array([energy.replace("?","nan") for energy in hbond["acceptor_1_energy"]], dtype=float).tolist()
        acceptor2_list = np.array([energy.replace("?","nan") for energy in hbond["acceptor_2_energy"]], dtype=float).tolist()
        donor1_list = np.array([energy.replace("?","nan") for energy in hbond["donor_1_energy"]], dtype=float).tolist()
        donor2_list = np.array([energy.replace("?","nan") for energy in hbond["donor_2_energy"]], dtype=float).tolist()
        
        for asym_id, seq_id, acceptor1, acceptor2, donor1, donor2 in zip(asym_id_list, seq_id_list, acceptor1_list, acceptor2_list, donor1_list, donor2_list):
            
            if not asym_id in hbond_dict:
                hbond_dict[asym_id] = {}
            if not seq_id in hbond_dict[asym_id]:
                hbond_dict[asym_id][seq_id] = {}

            hbond_dict[asym_id][seq_id]["acceptor_1_energy"] = acceptor1
            hbond_dict[asym_id][seq_id]["acceptor_2_energy"] = acceptor2
            hbond_dict[asym_id][seq_id]["donor_1_energy"] = donor1
            hbond_dict[asym_id][seq_id]["donor_2_energy"] = donor2
        
        return hbond_dict



class restraints:
    
    def get_module_path(self, filename):
        
        pth = f"module" 
        
        f = os.path.join(pth,filename)
        
        if Path(f).exists():
            return f
        else:
           sys.exit(f"{filename} does not exist")
    
    def get_restraints_path(self):

        restrain_path = self.get_module_path(f"restraints.json")

        return restrain_path
    
    def get_parameters_wh_path(self):

        parameters_wh_path = self.get_module_path(f"parameters_wh.json")

        return parameters_wh_path
    
    
    def get_restraints_dict(self):
        
        json_file = self. get_restraints_path()
        
        with open(json_file, "r") as f:
            
            file = json.loads(f.read())
            
        return file
    
    def get_parameters_wh_dict(self):
        
        json_file = self.get_parameters_wh_path()
        
        with open(json_file, "r") as f:
            
            file = json.loads(f.read())
            
        return file
        

   

    
