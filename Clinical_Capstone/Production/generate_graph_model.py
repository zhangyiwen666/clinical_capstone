from Production.utils.graph_model_utils import *
from scipy import sparse

def generate_matrix(G, nct2index, index2nct, n_step):
    """Return similarity matrix of given trials and their indices."""
    n = len(nct2index)
    similarity_matrix = np.zeros((n, n))
    for nct in nct2index:
        similar_units = find_similar_units(G, nct, n_step, 'NCT')
        for similar, step in similar_units.items():
            if similar in nct2index:
                i, j = nct2index[nct], nct2index[similar]
                similarity_matrix[i][j] = 1.2 - 0.2 * step
    return similarity_matrix

if __name__ == "__main__":
    from time import time
    t0 = time()
    # read data
    path = 'Data'
    nct2doctors = get_nct2doctors(False, '%s/Facility_Investigators.csv' % path)
    nct2citations = get_nct2citations('%s/Study_References.csv' % path)
    nct2conditions = get_nct2conditions('%s/Browse_Conditions.csv' % path)
    nct2interventions = get_nct2interventions('%s/Browse_Interventions.csv' % path)
     
    all_citations = get_all_items(nct2citations)
    all_conditions = get_all_items(nct2conditions)
    all_interventions = get_all_items(nct2interventions)
    
    cit2conditions = get_cit2conditions(all_citations, all_conditions)
    cit2interventions = get_cit2interventions(all_citations, all_interventions)

    year2nct, nct2year = read_nct_id_by_year('%s/ids_by_year_fda_reg_with_pos.pkl' % path)
    print("Finish reading data", time()-t0)
    t0 = time()
    
    # build graph with trials, doctors, citations and conditions
    #G = build_graph(nct2doctors, nct2citations, nct2conditions, cit2conditions, cit2authors, all_conditions)
    G = build_graph(nct2doctors, nct2citations, nct2conditions, cit2conditions, nct2interventions, cit2interventions)
    print("Finish building graph", time()-t0)
    t0 = time()
    
    # get similarity matrix for each year and save it as a .npz file
    year2matrix = {}
    for year in range(2014, 2019):
        nct2index = year2nct[year]
        index2nct = {idx: nct for nct, idx in nct2index.items()}
        similarity_matrix = generate_matrix(G, nct2index, index2nct, n_step=3)
        year2matrix[year] = sparse.csr_matrix(similarity_matrix)

    with open('Output/final_graph_model.pkl', 'wb') as f:
        pickle.dump(year2matrix, f)
    print("Finish creating matrices", time()-t0)


