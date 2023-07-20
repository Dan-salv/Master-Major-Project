#!/usr/bin/env bash

cat data/pdb_id/besttls_cif | parallel --jobs 40 --eta '/usr/local/bin/mkdssp --write-experimental /media/d.alvarez/T7/raw_data/density_fit/{}/{}_besttls.cif /media/d.alvarez/T7/raw_data/density_fit/{}/{}_besttls_dssp.cif' 





