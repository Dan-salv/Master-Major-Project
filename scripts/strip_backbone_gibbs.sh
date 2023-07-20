#!/usr/bin/env bash

pathDir=("/local_data/pdb_redo")
#pathDir="data/pdb_redo"


function backbone_only() {

in=$1;

out=$2;

mmCQL "${in}" "${out}" <<EOF
DELETE from atom_site WHERE (label_atom_id <> 'C' AND label_atom_id <> 'N' and label_atom_id <> 'CA' and label_atom_id <> 'O');
EOF
};

export -f backbone_only


#cat ./data/density_fit/pdb_id | parallel --jobs 40 --eta 'pdb2cif /local_data/pdb_redo/{=s/^.// | ~s/.{1}$// =}/{}/{}_besttls.pdb.gz ./data/density_fit/{}/{}_besttls.cif' 

#cat ./data/pdb_id_r | parallel --jobs 40 --eta 'backbone_only /local_data/pdb_redo/{=s/^.// | ~s/.{1}$// =}/{}/{}_final.cif ./data/density_fit/{}/{}_final_stripped.cif'

cat ./data/pdb_id/besttls/besttls_left | parallel --jobs 60 --timeout 108000 --eta 'backbone_only ./data/density_fit/{}/{}_besttls.cif ./data/density_fit/{}/{}_besttls_stripped.cif'

#backbone_only /local_data/pdb_redo/nj/1nji/1nji_final.cif 1nji_final_stripped.cif

