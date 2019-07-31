import pandas as pd
import numpy as np
from itertools import product
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def import_manual_validation(filepaths):
    """
    Import and store trial similarities manually scored

    input: list of filepaths
    """

    # Create list of pandas dataframes to be concatenated
    manual_validation_list = []
    for file in filepaths:
        manual_validation_list.append(pd.read_csv(file, header=None))

    # Concatenate manual validations
    manual_validation = pd.concat(manual_validation_list)

    # Copy validated and stack the reverse
    manual_validation_rev = manual_validation.copy()

    # Add column names
    manual_validation.columns = ['nct_id_x', 'nct_id_y', 'sim']
    manual_validation_rev.columns = ['nct_id_y', 'nct_id_x', 'sim']

    # Stack pairwise comparisons with its reverse to get symmetric validation
    manual_validation_concat = pd.concat([manual_validation, manual_validation_rev])

    # Average validated scores
    manual_validation_avg = pd.DataFrame(
        manual_validation_concat.groupby(['nct_id_x', 'nct_id_y'])['sim'].mean()).reset_index()

    return manual_validation_avg


def build_validation_matrix(filepaths,lookup_ids_index):
    """
    Build year year matrices for validated entries

    input: list of filepaths
    input: {year: {nct_id: matrix index}}
    """

    validated_similarities = import_manual_validation(filepaths)

    for year in range(2014,2019):
        index=lookup_ids_index[year]
        shape = (len(index), len(index))
        validationMatrix = -np.ones(shape)

        for _, row in validated_similarities.iterrows():
            try:
                validationMatrix[index[row['nct_id_x']]][index[row['nct_id_y']]] = row['sim']
            except:
                None

        if year == 2014:
            validationMatrix_ravel = np.ravel(validationMatrix)
        else:
            validationMatrix_ravel = np.hstack([validationMatrix_ravel,np.ravel(validationMatrix)])

    validated_indices = np.where(validationMatrix_ravel > -1)

    return validationMatrix_ravel[validated_indices], validated_indices


def build_covariates(model_similarities, validated_indices):
    """
    Transform set of model results into covariates for regression

    input: {"model name": {year: sparse similarity matrix}
    input: array of indices
    """
    X_pred = []
    X_train = []
    for model in model_similarities:
        X_pred.append(np.hstack([np.ravel(v.todense()) for k, v in model_similarities[model].items()]))
        X_train.append(
            np.hstack([np.ravel(v.todense()) for k, v in model_similarities[model].items()])[validated_indices])

    return np.vstack(X_pred).T, np.vstack(X_train).T


def predict_similarity(X_train, X_pred, y, feature_expansion='linear'):
    """
    Run the linear regression of estimated similarities and validated similarities

    input: covariates to train
    input: covariates to predict
    input: trained values
    """
    if feature_expansion == 'polynomial':
        print("Fitting polynomial expansion...")
        poly = PolynomialFeatures(degree=2, include_bias=False).fit(X_train)
        X_train = poly.transform(X_train)
        X_pred = poly.transform(X_pred)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y)
    y_pred = np.clip(reg.predict(X_pred),0,1)

    return y_pred


def gen_final_model(y_pred, lookup_ids_index):
    """
    Predict final model and create dictionary

    input: predicted similarity values
    input: {year: {nct_id: matrix index}}
    """

    reshape_amt = {year: len(lookup_ids_index[year])  for year in range(2014, 2019)}
    cumsum = 0
    reshape_indices = {}
    for year in range(2014, 2019):
        reshape_indices[year] = [cumsum, cumsum + reshape_amt[year]**2]
        cumsum += reshape_amt[year]**2

    final_models = {}
    for year in range(2014, 2019):
        final_models[year] = csr_matrix(y_pred[reshape_indices[year][0]:reshape_indices[year][1]].reshape(reshape_amt[year],reshape_amt[year]))

    return final_models


def random_validation(model, year, lookup_index_ids, num_ids=20, top_k=3):
    """
    Helper function for manual validation

    input: {year: sparse similarity matrix}
    input: year
    input: {year: {nct_id: matrix index}}
    input: number of random nct_ids to select
    input: number most similar matching pairs
    """
    matrix = np.array(model[year].todense())

    # Select n random nct_ids
    id_list = np.random.choice(matrix.shape[0], num_ids)
    print("*" * 54)

    # Loop over each random ID
    for nct_id in id_list:
        print("\nRandom Choice: https://clinicaltrials.gov/ct2/show/record/{}\n".format(lookup_index_ids[year][nct_id]))

        # Select the top 4
        top_list = np.array(matrix[nct_id,].argsort()[::-1][:top_k])

        # Loop over top 3
        for i in range(len(top_list)):
            print("Top {}: https://clinicaltrials.gov/ct2/show/record/{}".format(i + 1, lookup_index_ids[year][top_list[i]]))

        print("*" * 54)


def build_company_similarity(model,
                             year_list,
                             sponsor_by_year_with_pos_rev,
                             sponsor_2_nct_ids_by_year,
                             ids_by_year):
    """
    Build company-wise comparison matrix

    input: {year: sparse similarity matrix}
    input: list of years
    input: {year: {co matrix pos: clean sponsor}}
    input: {year: {sponsor: nct_id}}
    input: {year: {nct_id: trial matrix position}}
    """
    company_sim_mat_by_year = {year: np.zeros((len(sponsor_by_year_with_pos_rev[year]),
                                               len(sponsor_by_year_with_pos_rev[year]))) for year in year_list}


    #setting diagonal to 1

    for year in year_list:
        for i in range(company_sim_mat_by_year[year].shape[0]):
            company_sim_mat_by_year[year][i,i]=1


    #filling in non-diagonal elements

    for year in year_list:
        for i in range(len(sponsor_by_year_with_pos_rev[year])):
            for j in range(i):
                spons_a, spons_b = sponsor_by_year_with_pos_rev[year][i], sponsor_by_year_with_pos_rev[year][j]
                nct_id_tup_list = [elem for elem in product(sponsor_2_nct_ids_by_year[year][spons_a],
                                                            sponsor_2_nct_ids_by_year[year][spons_b])]
                avg_trial_sim = np.mean([model[year][ids_by_year[year][elem[0]],
                                                                 ids_by_year[year][elem[1]]] for elem in nct_id_tup_list])
                company_sim_mat_by_year[year][i,j] = avg_trial_sim
                company_sim_mat_by_year[year][j,i] = avg_trial_sim

    return company_sim_mat_by_year