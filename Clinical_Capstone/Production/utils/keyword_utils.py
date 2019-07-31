import pandas as pd
import re
import pickle
from scipy import sparse
from sklearn.preprocessing import normalize

###############################
######### IMPORT DATA #########
###############################


def import_nctid_list(filepath):
    """
    Returns a dictionary of index:NCT_ID and NCT_ID:index
    """
    with open(filepath, 'rb') as f:
        lookup_ids_index = pickle.load(f)

        lookup_index_ids = {}
    for year in lookup_ids_index:
        lookup_index_ids[year] = {v: k for k, v in lookup_ids_index[year].items()}

    return lookup_ids_index, lookup_index_ids


def import_keywords(file, df_parent_terms):
    """
    Import AACT keyword tables and generate dataframe that merges parent keywords as well
    """
    table = pd.read_csv(file)

    table.drop(['Unnamed: 0'], axis=1, inplace=True)

    table_parents = table.merge(df_parent_terms, on=['downcase_mesh_term'], how='left')
    table_parents.drop(['downcase_mesh_term'], axis=1, inplace=True)
    table_parents.columns = ['id', 'nct_id', 'mesh_term', 'downcase_mesh_term']

    return table, table_parents


def import_mesh_hierarchy(filepath):
    """
    Import external MeSH keyword hierarchy
    """
    with open(filepath, 'rb') as f:
        mesh = f.readlines()

    numbers = dict()
    terms = dict()

    for line in mesh:
        meshTerm = re.search(b'MH = (.+)$', line)
        if meshTerm:
            term = meshTerm.group(1).lower()

        meshNumber = re.search(b'MN = (.+)$', line)
        if meshNumber:
            number = meshNumber.group(1)

            numbers[number.decode('utf-8')] = term.decode('utf-8')

            # If term has more than one number, concatenate separated by space
            if term in terms:
                terms[term.decode('utf-8')] = terms[term.decode('utf-8')] + ' ' + number.decode('utf-8')
            else:
                terms[term.decode('utf-8')] = number.decode('utf-8')

    return numbers, terms


def build_parent_keyword_dict(lookup_number_keyword):
    """
    Creates dataframe of keyword to parent keywords
    """
    dict_parent_terms = {'downcase_mesh_term': [], 'parent_term': []}

    for id in lookup_number_keyword.keys():
        for i in range(len(id.split("."))):
            dict_parent_terms['downcase_mesh_term'].append(lookup_number_keyword[id])
            dict_parent_terms['parent_term'].append(lookup_number_keyword[id[:i*4+3]])

    df_parent_terms = pd.DataFrame.from_dict(dict_parent_terms).drop_duplicates()

    return df_parent_terms

###############################
### SIMILARITY CALCULATIONS ###
###############################


def sim_equality(df, index):
    """
    Matches number of keywords between sets of keywords
    """

    ids = pd.DataFrame([key for key in index.keys()], columns=['nct_id'])
    df = df.merge(ids, on=['nct_id'])


    df['key'] = 1
    df_xjoin = pd.merge(df, df, on=['key', 'downcase_mesh_term'])
    df_xjoin = df_xjoin.loc[(df_xjoin.nct_id_x != df_xjoin.nct_id_y)]

    df_term_counts = pd.DataFrame(df_xjoin.groupby(['nct_id_x', 'nct_id_y']).size()).reset_index()
    df_term_counts.columns = ['nct_id_x', 'nct_id_y', 'data']

    rows = df_term_counts.nct_id_x.apply(lambda x: index[x])
    cols = df_term_counts.nct_id_y.apply(lambda x: index[x])
    data = df_term_counts.data

    comparison = sparse.csr_matrix((data, (rows, cols)))

    return normalize(comparison, norm='max', axis=1)


def sim_jaccard(df, index):
    """
    Matches Jaccard distance between sets of keywords
    """

    def jaccard_calc(set1, set2):
        union = len(set1.union(set2))
        intersection = len(set1.intersection(set2))
        return intersection / union

    ids = pd.DataFrame([key for key in index.keys()], columns=['nct_id'])
    df = df.merge(ids, on=['nct_id'])
    df['key'] = 1

    df_xjoin_lim = pd.merge(df, df, on=['key', 'downcase_mesh_term'])[['nct_id_x', 'nct_id_y']].drop_duplicates()
    df_xjoin_lim = df_xjoin_lim.loc[(df_xjoin_lim.nct_id_x != df_xjoin_lim.nct_id_y)]

    df = pd.DataFrame(df.groupby('nct_id')['downcase_mesh_term'].apply(set).reset_index(),
                      columns=["nct_id", "downcase_mesh_term"])

    df['key'] = 1

    df_xjoin = pd.merge(df, df, on=['key'])
    df_xjoin = df_xjoin.merge(df_xjoin_lim, on=['nct_id_x', 'nct_id_y'], how='inner')
    df_xjoin['data'] = df_xjoin.apply(lambda x: jaccard_calc(x['downcase_mesh_term_x'], x['downcase_mesh_term_y']),
                                      axis=1)

    rows = df_xjoin.loc[(df_xjoin.data > 0)].nct_id_x.apply(lambda x: index[x])
    cols = df_xjoin.loc[(df_xjoin.data > 0)].nct_id_y.apply(lambda x: index[x])
    data = df_xjoin.loc[(df_xjoin.data > 0)].data

    comparison = sparse.csr_matrix((data, (rows, cols)))

    return comparison
