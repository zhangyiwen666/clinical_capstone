import numpy
import pandas
import string
from textblob import Word
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def nlp_pipeline_tfidf(
  series, 
  stop_list):
    
    """
    Returns a processed Pandas Series after stripping whitespace
    characters, removing non-alphanumeric characters, converting
    to lowercase, removing stop words and punctuation, stemming 
    and lemmatizing

    Input: Unprocessed Pandas Series (text column)
    """

    series = series.apply(lambda x: [elem.strip() for elem in x])
    series = series.apply(lambda x: [elem for elem in x if not elem.isdigit()])
    series = series.apply(lambda x: [elem.lower() for elem in x])
    series = series.apply(lambda x: [elem for elem in x if elem not in stop_list])
    series = series.apply(lambda x: [elem.replace('[^\w\s]','') for elem in x])
    st = PorterStemmer()
    series = series.apply(lambda x: [st.stem(elem) for elem in x])
    series = series.apply(lambda x: [Word(elem).lemmatize() for elem in x])
    series = series.apply(lambda x: ' '.join(x))
    return series

def display_scores(
    vectorizer, 
    tfidf_result, 
    size):

    """
    Display names and scores of top TF-iDF features

    Input: 'vectorizer' - TF-iDF object
    Input: 'tfidf_result' - n_docs*vocab_size matrix of TF-iDF vectors
    Input: 'size' - # top features to display
    """

    scores = zip(vectorizer.get_feature_names(),
                 numpy.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores[:size]:
        print("{0:50} Score: {1}".format(item[0], item[1]))

def gen_similarity(
    data_frame,
    desc_col,
    indices_by_year,
    stop_list,
    desc_col_type = 'one', 
    max_features=300,
    year_list = list(range(2014,2019)),
    ngram_range=(1,3),
    min_df=0.05,
    max_df=0.95):

    """
    Returns dictionary of TF-iDF vectorizer objects (per year), 
    dictionary of TF-iDF vectors (per year) and dictionary of
    trial similarity matrices (per year) 

    Input: 'data_frame' - Table of AACT Database which contains text field of interest
    Input: 'desc_col' - Name of Textual field of interest
    Input: 'indices_by_year' - Dictionary of NCT IDs and their corresponding positions (per year)
    Input: 'desc_col_type' - Whether one or many rows per NCT ID in field of interest
    Input: 'max_features' - Upper bound to vocabulary of fit TF-iDF vectorizer
    Input: 'year_list' - trial submission period to be considered
    Input: 'ngram_range' - ngrams or associations of words to be considered
    Input: 'min_df' - minimum representation threshold of word across all documents
    Input: 'max_df' - maximum representation threshold of word across all documents
    """

    desc_sel_by_year = {}
    actual_info_by_year = {}
    
    for year in year_list:
        desc_sel_by_year[year] = pandas.DataFrame(list(indices_by_year[year].keys()),
                                              columns = ["nct_id"])
        actual_info_by_year[year] = \
        data_frame[data_frame['nct_id'].isin(indices_by_year[year])][['nct_id',desc_col]].copy()
        desc_sel_by_year[year] = desc_sel_by_year[year].merge(actual_info_by_year[year],
                                                              on='nct_id',
                                                              how='left')
    
    if desc_col_type == 'many':
        desc_by_nct_id_by_year = \
        {key:{key_2:"" for key_2,val_2 in val.items()} for key,val in indices_by_year.items()}

        for year in year_list:
            for idx in desc_by_nct_id_by_year[year]:
                desc_by_nct_id_by_year[year][idx]+=\
                actual_info_by_year[year][actual_info_by_year[year]['nct_id']==idx][desc_col].values
        
        desc_sel_by_year = {year: pandas.DataFrame(list(desc_by_nct_id_by_year[year].items()), 
                                               columns= ['nct_id',desc_col]) for year in year_list}

    tfidf_list = {}
    vect_list = {}
    similarities = {}
    
    for year in year_list:
        desc_sel_by_year[year].set_index('nct_id',inplace = True)
        desc_sel_by_year[year][desc_col].fillna('',inplace = True)
        
        if desc_col_type == 'one':
            desc_sel_by_year[year][desc_col] = \
            desc_sel_by_year[year][desc_col].apply(lambda x: x.strip()).apply(lambda x: x.split(' '))
        
        desc_sel_by_year[year][desc_col] = nlp_pipeline_tfidf(desc_sel_by_year[year][desc_col], 
                                                              stop_list)
        desc_sel_by_year[year] = \
        desc_sel_by_year[year].loc[[elem[0] for elem in sorted(indices_by_year[year].items(), 
                                                               key=lambda kv:kv[1])]]
        idx_order = list(desc_sel_by_year[year].index)
        
        tfidf_list[year] = TfidfVectorizer(max_features=max_features, 
                                           lowercase=True, 
                                           analyzer='word', 
                                           stop_words= stop_list ,
                                           min_df= min_df,
                                           max_df = max_df,
                                           ngram_range=ngram_range)
        
        vect_list[year] = tfidf_list[year].fit_transform(desc_sel_by_year[year][desc_col].values)
        
        similarities[year] = pandas.DataFrame(data=(cosine_similarity(vect_list[year])),
                                          index=idx_order,
                                          columns = idx_order
                                         )
    
    return tfidf_list,vect_list,similarities

def nlp_pipeline_word2vec(
  series,
  stop_list):
    
    """
    Returns a processed Pandas Series after stripping whitespace
    characters, converting to lowercase, tokenizing words, and 
    removing stop words and punctuation

    Input: Unprocessed Pandas Series (text column)
    """

    series = series.apply(lambda x: x.strip().lower().split(' '))
    series = series.apply(lambda x: [elem.strip(string.punctuation) for elem in x])
    series = series.apply(lambda x: \
      [elem.strip() for elem in x if (elem not in stop_list) and (elem!='')])
    return series

def gen_vec_from_list(
  str_list,
  model):

    """
    Returns a 200 dimensional dense vector corresponding to 'str_list'

    Input: 'str_list' - row of tokenized and processed text-based Pandas series
    Input: 'model' - pre-trained (preferably on medical corpus) word2vec model
    """

    vec_list=[]
    elem_list = []
    for elem in str_list:
        try:
            curr_vec = model[elem]
            vec_list.append(curr_vec)
            elem_list.append(elem)
        except:
            continue
    return pandas.DataFrame(data=vec_list,
                        index=elem_list).apply(numpy.mean,
                                               axis=0).values

def random_validation_text(
  model, 
  year, 
  lookup_index_ids, 
  num_ids=20, 
  top_k=3):
    """
    Helper function for manual validation

    input: {year: sparse similarity matrix}
    input: year
    input: {year: {nct_id: matrix index}}
    input: number of random nct_ids to select
    input: number most similar matching pairs
    """
    matrix = model[year]
    # Select n random nct_ids
    id_list = numpy.random.choice(matrix.shape[0], 
                                  num_ids)
    print(id_list)
    print("*" * 54)

    # Loop over each random ID
    for idx in id_list:
        print("\nRandom Choice: https://clinicaltrials.gov/ct2/show/record/{}\n".format(lookup_index_ids[year][idx]))

        # Select the top 4
        top_list = numpy.array(matrix.values[idx,].argsort()[::-1][:top_k])

        # Loop over top 3
        for i in range(len(top_list)):
            print("Top {}: https://clinicaltrials.gov/ct2/show/record/{}".format(i + 1, lookup_index_ids[year][top_list[i]]))

        print("*" * 54)

def build_covariates_text(
  model_similarities, 
  validated_indices):
    """
    Transform set of model results into covariates for regression

    input: {"model name": {year: sparse similarity matrix}
    input: array of indices
    """
    X_pred = []
    X_train = []
    for model in model_similarities:
        X_pred.append(numpy.hstack([numpy.ravel(v.values) for k, v in model_similarities[model].items()]))
        X_train.append(
            numpy.hstack([numpy.ravel(v.values) for k, v in model_similarities[model].items()])[validated_indices])

    return numpy.vstack(X_pred).T, numpy.vstack(X_train).T