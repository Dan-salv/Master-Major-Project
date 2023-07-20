#!/usr/bin/env bash

time cat ./scripts/bash/pdb_id | parallel --jobs 40 --eta 'python3 scripts/py/random_rsccs.py -i {} -o /DATA/raw_data/density_fit'

python3 scripts/py/merge_dataframe.py -d /DATA/raw_data/density_fit -i random_no_filter -o data/training_data/all_random_conformations_no_filter




