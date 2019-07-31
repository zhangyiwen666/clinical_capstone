import time
from Production.utils.utils import *
from Production.utils.keyword_utils import *

###############################
######### IMPORT DATA #########
###############################

print("Import data...")

# Import set of nct_ids for the final matrices
lookup_ids_index, lookup_index_ids = import_nctid_list('Data/ids_by_year_fda_reg_with_pos.pkl')

# Import MeSH keywords
lookup_number_keyword, lookup_keyword_number= import_mesh_hierarchy('Data/d2018.bin')
df_parent_terms = build_parent_keyword_dict(lookup_number_keyword)

# Import keyword files and expand parent MeSH terms
browse_conditions, browse_conditions_parents = import_keywords('Data/Browse_Conditions.csv', df_parent_terms)
browse_interventions, browse_interventions_parents = import_keywords('Data/Browse_Interventions.csv', df_parent_terms)


###############################
############ MODELS ###########
###############################

print("Generate models...")


def gen_sim_matrices(df,similarity_metric,lookup_ids_index=lookup_ids_index):
    """
    Run similarities
    """
    print(f"\n***** {df} {similarity_metric} *****")
    model = {}
    for year in sorted(lookup_ids_index.keys()):
        if year >= 2014:
            start = time.time()
            model[year] = eval(similarity_metric)(eval(df),lookup_ids_index[year])
            print(f'Ran {year} in {time.time()-start:.1f} seconds')

    return model

model_similarities = {
    'm1_conditions':    gen_sim_matrices('browse_conditions',            'sim_equality'),
    'm1_interventions': gen_sim_matrices('browse_interventions',         'sim_equality'),
    'm2_conditions':    gen_sim_matrices('browse_conditions_parents',    'sim_equality'),
    'm2_interventions': gen_sim_matrices('browse_interventions_parents', 'sim_equality'),
    'm3_conditions':    gen_sim_matrices('browse_conditions',            'sim_jaccard'),
    'm3_interventions': gen_sim_matrices('browse_interventions',         'sim_jaccard'),
    'm4_conditions':    gen_sim_matrices('browse_conditions_parents',    'sim_jaccard'),
    'm4_interventions': gen_sim_matrices('browse_interventions_parents', 'sim_jaccard')
}

###############################
###### MANUAL VALIDATION ######
###############################

random_validation(model_similarities['m1_conditions'], 2018, lookup_index_ids)


###############################
####### ENSEMBLE MODEL ########
###############################

# Filepath for validation set
filepaths = ['Manual Validation/manual validation - ba.csv',
             'Manual Validation/manual validation - xmyz.csv',
             'Manual Validation/manual validation - ashs.csv']

# Build validation matrix
y, validated_indices = build_validation_matrix(filepaths, lookup_ids_index)

# Prepare covariates
X_pred, X_train = build_covariates(model_similarities, validated_indices)

# Predict similarities, add interaction to regression covariates
y_pred = predict_similarity(X_train, X_pred, y, feature_expansion='polynomial')

# Separate predictions into year matrices
final_models = gen_final_model(y_pred, lookup_ids_index)

print("Saving model...")

# Save final keyword models
with open('Output/final_keyword_model.pkl', 'wb') as f:
    pickle.dump(final_models, f)
