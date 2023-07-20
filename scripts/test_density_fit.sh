#!/usr/bin/env bash

pathDir=("/local_data/pdb_redo")
#pathDir="data/pdb_redo"


density-fitness \
--use-auth-ids \
--sampling-rate 1.5 \
--hklin /DATA/pdb_redo/ia/1iat/1iat_final.mtz \
--xyzin /DATA/raw_data/density_fit/1iat/1iat_final_stripped.cif \
--no-edia \
-o /DATA/raw_data/density_fit/1iat/1iat_final.json \
--output-format json \

