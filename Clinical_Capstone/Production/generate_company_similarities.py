import pickle
from Production.utils.utils import *

########################################
########### READ INDEX FILES ###########
########################################


sponsor_2_nct_ids_by_year = pickle.load(open("Data/sponsor_2_nct_ids_by_year.pkl", "rb"))
ids_by_year = pickle.load(open("Data/ids_by_year_fda_reg_with_pos.pkl", "rb"))

year_list = list(range(2014,2019))

sponsor_by_year_with_pos = {year:{elem:i for i,elem in enumerate(sorted(list(sponsor_2_nct_ids_by_year[year].keys())))} for year in year_list}
sponsor_by_year_with_pos_rev = {year:{i:elem for i,elem in enumerate(sorted(list(sponsor_2_nct_ids_by_year[year].keys())))} for year in year_list}


###################################
########### LOAD MODELS ###########
###################################

final_keyword_models = pickle.load(open("Output/final_keyword_model.pkl","rb"))
final_graph_models = pickle.load(open("Output/final_graph_model.pkl","rb"))
final_text_models = pickle.load(open("Output/final_text_model.pkl","rb"))

for year in final_text_models:
    final_text_models[year] = final_text_models[year].todense()
    final_keyword_models[year] = final_keyword_models[year].todense()
    final_graph_models[year] = final_graph_models[year].todense()


###################################
###### GENERATE COMPANY SIMS ######
###################################


final_keyword_company_similarity_model = build_company_similarity(final_keyword_models,
                                                                  year_list,
                                                                  sponsor_by_year_with_pos_rev,
                                                                  sponsor_2_nct_ids_by_year,
                                                                  ids_by_year)

final_graph_company_similarity_model = build_company_similarity(final_graph_models,
                                                                  year_list,
                                                                  sponsor_by_year_with_pos_rev,
                                                                  sponsor_2_nct_ids_by_year,
                                                                  ids_by_year)

final_text_company_similarity_model = build_company_similarity(final_text_models,
                                                               year_list,
                                                               sponsor_by_year_with_pos_rev,
                                                               sponsor_2_nct_ids_by_year,
                                                               ids_by_year)

###############################
###### SAVE COMPANY SIMS ######
###############################

pickle.dump(final_keyword_company_similarity_model,
            open("Output/final_keyword_company_similarity_model.pkl","wb"),
            protocol=2)

pickle.dump(final_graph_company_similarity_model,
            open("Output/final_graph_company_similarity_model.pkl","wb"),
            protocol=2)

pickle.dump(final_text_company_similarity_model,
            open("Output/final_text_company_similarity_model.pkl","wb"),
            protocol=2)
