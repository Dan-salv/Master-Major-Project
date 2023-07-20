#!/usr/bin/env bash

cat data/pdb_id/pdb_id | parallel --jobs 40 --eta 'tortoize /DATA/raw_data/density_fit/{}/{}_besttls.cif /DATA/raw_data/density_fit/{}/{}_besttls_tortoize.json' 





