#!/usr/bin/env bash

pathDir=("/local_data/pdb_redo")
#pathDir="data/pdb_redo"


# density-fitness \
# --use-auth-ids \
# --sampling-rate 1.5 \
# --hklin /DATA/pdb_redo/a0/1a0b/1a0b_final.mtz \
# --xyzin /DATA/raw_data/density_fit/1a0b/1a0b_final_stripped.cif \
# --no-edia \
# --output /DATA/raw_data/density_fit/1a0b/1a0b_final.json \
# --output-format json \

# cat data/pdb_id/pdb_id | parallel --jobs 40 --eta 'density-fitness \
# --use-auth-ids \
# --sampling-rate 1.5 \
# --hklin /DATA/pdb_redo/{=s/^.// | ~s/.{1}$// =}/{}/{}_final.mtz \
# --xyzin /DATA/raw_data/density_fit/{}/{}_final_stripped.cif \
# --no-edia \
# --output  /DATA/raw_data/density_fit/{}/{}_final.json \
# --output-format json \
# '
# echo 'obtained json files from final models'

cat data/pdb_id/pdb_id | parallel --jobs 40 --eta 'density-fitness \
--use-auth-ids \
--sampling-rate 1.5 \
--hklin /DATA/pdb_redo/{=s/^.// | ~s/.{1}$// =}/{}/{}_final.mtz \
--xyzin /DATA/raw_data/density_fit/{}/{}_besttls_stripped.cif \
--no-edia \
--output  /DATA/raw_data/density_fit/{}/{}_besttls.json \
--output-format json \
'
echo 'obtained json files from besttls models'


