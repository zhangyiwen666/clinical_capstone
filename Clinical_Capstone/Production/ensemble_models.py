import json
import pandas as pd
import psycopg2
import numpy as np
import re
import pickle
import time
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression


#############################
####### IMPORT MODELS #######
#############################

filepaths = ['../Output/final_keyword_model.pkl']

models = []
for filepath in filepaths:
    with open(filepath, 'rb') as f:
        models.append(pickle.load(f))


###########################
#### IMPORT VALIDATION ####
###########################

def import_manual_validation(filepaths):
    """
    Import and store trial similarities manually scored
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

filepaths = [''../Manual Validation/manual validation - ba.csv',
             '../Manual Validation/manual validation - xmyz.csv']

validated_similarities = import_manual_validation(filepaths)


def build_validation_matrix(validated_similarities=validated_similarities,lookup_ids_index=lookup_ids_index):
    """
    Build year year matrices for validated entries
    """

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


y, validated_indices =  build_validation_matrix()


############################
#### PREPARE COVARIATES ####
############################

def build_covarietes(models=models, validated_indices=validated_indices):
    """
    Transform set of model results into covariates for regression
    """

    X_pred = []
    X_train = []
    for model in models:
        X_pred.append(np.hstack([np.ravel(v.todense()) for k, v in model.items()]))
        X_train.append(
            np.hstack([np.ravel(v.todense()) for k, v in model.items()])[validated_indices])

    return np.vstack(X_pred).T, np.vstack(X_train).T

X_pred, X_train = build_covarietes()


#############################
#### FIND MODEL WEIGHTS ####
#############################

def predict_similarity(X_train=X_train, X_pred=X_pred, y=y):
    """
    Run the linear regression of estimated similarities and validated similarities
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_train, y)
    y_pred = np.clip(reg.predict(X_pred),0,1)

    return y_pred

y_pred = predict_similarity()

#############################
##### BUILD FINAL MODEL #####
#############################

def gen_final_model(y_pred=y_pred, lookup_ids_index = lookup_ids_index):
    """
    Predict final model and create dictionary
    """

    reshape_amt = {year: len(lookup_ids_index[year])  for year in range(2014, 2019)}
    cumsum = 0
    reshape_indices = {}
    for year in range(2014, 2019):
        reshape_indices[year] = [cumsum, cumsum + reshape_amt[year]**2]
        cumsum += reshape_amt[year]**2

    final_models = {}
    for year in range(2014, 2019):
        print(reshape_indices[year])
        final_models[year] = csr_matrix(y_pred[reshape_indices[year][0]:reshape_indices[year][1]].reshape(reshape_amt[year],reshape_amt[year]))


    return final_models

final_models = gen_final_model()
final_models
