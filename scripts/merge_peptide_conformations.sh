#!/usr/bin/env bash

#python3 scripts/py/merge_dataframe.py -d /DATA/raw_data/density_fit -i peptide_conformation_no_filter -o data/training_data/all_peptide_conformations_no_filter

#python3 scripts/py/merge_dataframe.py -d /DATA/raw_data/density_fit -i peptide_conformation -o data/training_data/all_peptide_conformations

#python3 scripts/py/merge_dataframe.py -d /DATA/raw_data/density_fit -i random_filter -o data/training_data/all_random_conformations

#python3 scripts/py/merge_dataframe.py -d /DATA/raw_data/density_fit -i random_no_filter -o data/training_data/all_random_conformations_no_filter

python3 scripts/py/merge_dataframe.py -d /media/d.alvarez/T7/raw_data/density_fit/ -i training_peptide_data -o data/training_data/training_set

