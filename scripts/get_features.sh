#!/usr/bin/env bash

#cat data/pdb_id/besttls_cif | parallel --jobs 40 --eta '/usr/local/bin/mkdssp --write-experimental /media/d.alvarez/T7/raw_data/density_fit/{}/{}_besttls.cif /media/d.alvarez/T7/raw_data/density_fit/{}/{}_besttls_dssp.cif' 

cat data/pdb_id/pdb_id | parallel --jobs 40 --eta 'python3 scripts/py/get_features.py -i {} -d data/training_data -o /media/d.alvarez/T7/raw_data/density_fit/{}/peptide_conformation'

python3 scripts/py/merge_dataframe.py -d /media/d.alvarez/T7/raw_data/density_fit/ -i training_peptide_data -o data/training_data/training_set

