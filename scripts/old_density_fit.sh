#cat data/pdb_id/pdb_id | parallel 'echo  --jobs 40 --eta density-fitness \
#--use-auth-ids \
#--sampling-rate 1.5 \
#--hklin /DATA/pdb_redo/{=s/^.// | ~s/.{1}$// =}/{}/{}_final.mtz \
#--xyzin /DATA/raw_data/density_fit/{}/{}_besttls_stripped.cif \
#--no-edia \
#-o DATA/raw_data/density_fit/{}/{}_besttls.json \
#--output-format json \
#'

#time find $pathDir -name "*_final.mtz" | parallel --jobs 40 --eta  '/local_data/density-fitness \
#--use-auth-ids \
#--sampling-rate 1.5 \
#--hklin /local_data/pdb_redo/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}_final.mtz \
#--xyzin /local_data/pdb_redo/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}_besttls.pdb.gz \
#--no-edia \
#-o ./data/density_fit/besttls/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}_{= s:.*/([^/]*)/[^/]*:\1: =}_besttls.json \
#--output-format json \
#' 

#echo 'obtained json files from besttls models'

#time find $pathDir -name "*_final.mtz" | parallel --jobs 40 --eta '/local_data/density-fitness \
#--use-auth-ids \
#--sampling-rate 1.5 \
#--hklin /local_data/pdb_redo/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}_final.mtz \
#--xyzin /local_data/pdb_redo/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}_final.pdb \
#--no-edia \
#-o ./data/density_fit/final/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}_{= s:.*/([^/]*)/[^/]*:\1: =}_final.json \
#--output-format json \
#'




#{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: 	


#cat ./data/density_fit/pdb_id | parallel --jobs 40 --eta  '/local_data/density-fitness \
#--use-auth-ids \
#--sampling-rate 1.5 \
#--hklin /local_data/pdb_redo/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}_final.mtz \
#--xyzin /local_data/pdb_redo/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}/{= s:.*/([^/]*)/[^/]*:\1: =}_besttls.pdb.gz \
#--no-edia \
#-o ./data/density_fit/besttls/{= s:.*/([^/]*)/[^/]*/[^/]*:\1: =}_{= s:.*/([^/]*)/[^/]*:\1: =}_besttls.json \
#--output-format json \
#'
 
#density-fitness \
#--use-auth-ids \
#--sampling-rate 1.5 \
#--hklin /DATA/pdb_redo/a0/1a0b/1a0b_final.mtz \
#--xyzin ./data/density_fit/1a0b/1a0b_final_stripped.cif \
#--no-edia \
#-o ./data/prueba_strip \
#--output-format json \
