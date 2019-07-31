#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:26:11 2018

@author: yiwenzhang
"""

import sys, psycopg2
import json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Production.utils.text_utils import *
from Production.utils.utils import *
from Production.utils.keyword_utils import *

import nltk
from nltk.corpus import stopwords
from textblob import Word
import wget, gensim
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


# Import set of nct_ids for the final matrices
lookup_ids_index, lookup_index_ids = import_nctid_list('Data/ids_by_year_fda_reg_with_pos.pkl')


############################################################################
### creating stop-word list
############################################################################
stop_list = stopwords.words('english')
stop_list.extend(("a", "an","the","with","ii","iii","iv","non",
                  "studies","use","study","multiple", "single", "double",
                  "this","these","those","care","effect","health","patients",
                  "trial","treatment","versus","clinical","clinic","controlled","control"))


############################################################################
### connecting with Postgresql AACT Database and loading chosen NCT IDs
############################################################################

with open('Data/config.json') as f:
    conf = json.load(f)
    
conn_str = "host={} dbname={} port={} user={} password={}".format(conf['host'],
                                                                  conf['dbname'],
                                                                  conf['port'],
                                                                  conf['user'],
                                                                  conf['password'])
conn = psycopg2.connect(conn_str)


############################################################################
### Loading required tables
############################################################################

studies = pd.read_sql('select * from studies', 
                      con=conn)
keywords = pd.read_sql('select * from keywords', 
                       con=conn)
detailed_descriptions = pd.read_sql('select * from detailed_descriptions', 
                                    con=conn)


############################################################################
### Processing to consider only the NCT IDs of interest
############################################################################

studies = studies[studies['is_fda_regulated_drug']==True].copy()
studies['study_first_submitted_date'] = pd.to_datetime(studies['study_first_submitted_date'],
                                                       format='%Y-%m-%d')
studies = studies[studies['study_first_submitted_date'].dt.year.isin(range(2014,2019))]
chosen_ids = studies['nct_id'].tolist()

keywords = keywords[keywords['nct_id'].isin(chosen_ids)].copy()
detailed_descriptions = \
detailed_descriptions[detailed_descriptions['nct_id'].isin(chosen_ids)].copy()



############################################################################
### Generating TF-iDF objects, TF-iDF Vectors and Similarity Matrices
############################################################################

off_t_tfidf_list, off_t_vect_list, off_t_similarities = \
                                    gen_similarity(data_frame=studies,
                                                   desc_col='official_title',
                                                   indices_by_year = lookup_ids_index,
                                                   stop_list=stop_list,
                                                   desc_col_type = 'one',
                                                   max_features=300,
                                                   year_list = list(range(2014,2019)),
                                                   ngram_range=(1,3))

keyword_tfidf_list, keyword_vect_list, keyword_similarities = \
                                    gen_similarity(data_frame=keywords,
                                                 desc_col='downcase_name',
                                                 indices_by_year = lookup_ids_index,
                                                 stop_list=stop_list,
                                                 desc_col_type = 'many',
                                                 max_features=300,
                                                 year_list = list(range(2014,2019)),
                                                 ngram_range=(1,1),
                                                 min_df=0.01,
                                                 max_df=0.99)


ddesc_tfidf_list, ddesc_vect_list, ddesc_similarities = \
                                    gen_similarity(data_frame=detailed_descriptions,
                                                   desc_col='description',
                                                   indices_by_year = lookup_ids_index,
                                                   stop_list=stop_list,
                                                   desc_col_type = 'one',
                                                   max_features=300,
                                                   year_list = list(range(2014,2019)),
                                                   ngram_range=(1,3),
                                                   min_df=0.01, 
                                                   max_df=0.99)


############################################################################
### Downloading word2vec model and saving in working directory
############################################################################
                                    
word2vec_model_url = 'http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin'  
wget.download(word2vec_model_url, 'Data/PubMed-w2v.bin')


############################################################################
### loading downloaded word2vec model using gensim
############################################################################
model = gensim.models.KeyedVectors.load_word2vec_format('Data/PubMed-w2v.bin', 
                                                        binary=True)


############################################################################
### Generating word2vec based similarity matrices for Official Title
############################################################################

official_title_series = studies['official_title'].copy()
official_title_series.fillna('',inplace=True)
official_title_series = nlp_pipeline_word2vec(series=official_title_series,
                                              stop_list=stop_list)

official_title_vec_series = \
official_title_series.apply(lambda x: gen_vec_from_list(str_list=x,
                                                        model=model))

#addressing null vectors
official_title_vec_series = \
official_title_vec_series.apply(lambda x: np.array(np.zeros(shape=(200,))) if len(x)==0 else x)

official_title_vec_df = pd.DataFrame.from_items(zip(official_title_vec_series.index, 
                                                    official_title_vec_series.values)).T

official_title_series.index = chosen_ids
official_title_vec_series.index = chosen_ids
official_title_vec_df.index = chosen_ids

id_list_ordered_by_year = {year:[elem[0] for elem in sorted(lookup_ids_index[year].items(),
                                                            key=lambda kv:kv[1])] \
                            for year in range(2014,2019)}

year_list = list(range(2014,2019))

word2vec_off_tit_sim_mat_by_year = {}
for year in year_list:
    word2vec_off_tit_sim_mat_by_year[year] = \
                pd.DataFrame(data=cosine_similarity(official_title_vec_df.loc[id_list_ordered_by_year[year]]),
                              index=id_list_ordered_by_year[year],
                              columns=id_list_ordered_by_year[year])



############################################################################
### Generating word2vec based similarity matrices for Keywords
############################################################################

keywords_sel = \
keywords.groupby('nct_id').apply(lambda x: x['downcase_name'].tolist()).reset_index()
keywords_sel.columns = ['nct_id','keywords']
keywords_sel['keywords'] = keywords_sel['keywords'].apply(lambda x: ' '.join(x))

keywords_sel.set_index('nct_id',
                       inplace=True)
keywords_sel = pd.Series(data=keywords_sel['keywords'].values,
                         index=keywords_sel.index)

keywords_sel = nlp_pipeline_word2vec(series=keywords_sel,
                                     stop_list=stop_list)

keywords_sel = keywords_sel.apply(lambda x: gen_vec_from_list(str_list=x,
                                                              model=model))

all_ids = []
for year in year_list:
    all_ids += id_list_ordered_by_year[year]

keywords_sel_all_ids = pd.concat(objs=[pd.Series(data=all_ids,
                                                 index=all_ids),
                                       keywords_sel],
                                 axis=1)

keywords_sel_all_ids.drop(labels=[0],
                          axis=1,
                          inplace=True)

#addressing null vectors
keywords_sel_all_ids[1] = \
keywords_sel_all_ids[1].apply(lambda x: np.array(np.zeros(shape=(200,))) \
                              if ((isinstance(x,float)) or (len(x)==0)) else x)

keywords_vec_df = pd.DataFrame.from_items(zip(keywords_sel_all_ids.index, 
                                              keywords_sel_all_ids[1].values)).T

word2vec_keywords_sim_mat_by_year = {}
for year in year_list:
    word2vec_keywords_sim_mat_by_year[year] = \
    pd.DataFrame(data=cosine_similarity(keywords_vec_df.loc[id_list_ordered_by_year[year]]),
                  index=id_list_ordered_by_year[year],
                  columns=id_list_ordered_by_year[year])


############################################################################
### Generating word2vec based similarity matrices for Detailed Descriptions
############################################################################

detailed_descriptions_sel = pd.Series(data=detailed_descriptions['description'].values,
                                      index=detailed_descriptions['nct_id'].tolist())

detailed_descriptions_sel = nlp_pipeline_word2vec(series=detailed_descriptions_sel,
                                         stop_list=stop_list)

detailed_descriptions_sel = \
            detailed_descriptions_sel.apply(lambda x: gen_vec_from_list(str_list=x,
                                            model=model))

detailed_desc_sel_all_ids = pd.concat(objs=[pd.Series(data=all_ids,
                                                     index=all_ids),
                                       detailed_descriptions_sel],
                                      axis=1)
            
detailed_desc_sel_all_ids.drop(labels=[0],
                              axis=1,
                              inplace=True)

#addressing null vectors
detailed_desc_sel_all_ids[1] = \
detailed_desc_sel_all_ids[1].apply(lambda x: np.array(np.zeros(shape=(200,))) \
                                   if ((isinstance(x,float)) or (len(x)==0)) else x)

detailed_desc_vec_df = pd.DataFrame.from_items(zip(detailed_desc_sel_all_ids.index, 
                                                   detailed_desc_sel_all_ids[1].values)).T

word2vec_detailed_desc_sim_mat_by_year = {}
for year in year_list:
    word2vec_detailed_desc_sim_mat_by_year[year] = \
            pd.DataFrame(data=cosine_similarity(detailed_desc_vec_df.loc[id_list_ordered_by_year[year]]),
                      index=id_list_ordered_by_year[year],
                      columns=id_list_ordered_by_year[year])


############################################################################
### Combining all Model Similarities into one Dictionary
############################################################################

model_similarities = {
    'off_t_tfidf':              off_t_similarities,
    'off_t_word2vec':           word2vec_off_tit_sim_mat_by_year,
    'keyword_tfidf':            keyword_similarities,
    'keyword_word2vec':         word2vec_keywords_sim_mat_by_year,
    'd_desc_tfidf':             ddesc_similarities,
    'd_desc_word2vec':          word2vec_detailed_desc_sim_mat_by_year
}


###############################
### Manual validation 
###############################

random_validation_text(model_similarities['off_t_tfidf'], 
                       2018, 
                       lookup_index_ids)


###############################
### Ensemble Model 
###############################

# Filepath for validation set
filepaths = ['Manual Validation/manual validation - ba.csv',
             'Manual Validation/manual validation - xmyz.csv',
             'Manual Validation/manual validation - ashs.csv']

# Build validation matrix
y, validated_indices = build_validation_matrix(filepaths, lookup_ids_index)

# Prepare covariates
X_pred, X_train = build_covariates_text(model_similarities, validated_indices)

# Predict similarities, add interaction to regression covariates
y_pred = predict_similarity(X_train, X_pred, y, feature_expansion='polynomial')

# Separate predictions into year matrices
final_models = gen_final_model(y_pred, lookup_ids_index)

print("Saving model...")

# Save final text models
with open('Output/final_text_model.pkl', 'wb') as f:
    pickle.dump(final_models, f)