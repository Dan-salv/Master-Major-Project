#!/usr/bin/env bash


time cat ./data/pdb_id/pdb_id | parallel --jobs 44 --eta 'python3 scripts/py/filter_rsccs.py -i {} -o /DATA/raw_data/density_fit'




