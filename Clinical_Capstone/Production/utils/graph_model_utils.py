import pandas as pd
import pickle
import numpy as np
import networkx as nx

def get_nct2doctors(overall_officials=False, facility_investigators=False):
    """Return a dictionary: nct2doctors."""
    nct2doctors = {}
    
    if overall_officials: 
        df = pd.read_csv(overall_officials)
        for i in range(len(df)):
            nct_id, name = str(df.loc[i]['nct_id']), str(df.loc[i]['name'])
            if nct_id == 'nan' or name == 'nan':
                continue
            if nct_id in nct2doctors:
                nct2doctors[nct_id].add(name)
            else:
                nct2doctors[nct_id] = {name}
            
    if facility_investigators:
        df = pd.read_csv(facility_investigators)
        for i in range(len(df)):
            nct_id, name = str(df.loc[i]['nct_id']), str(df.loc[i]['name'])
            if nct_id == 'nan' or name == 'nan':
                continue
            if nct_id in nct2doctors:
                nct2doctors[nct_id].add(name)
            else:
                nct2doctors[nct_id] = {name} 
             
    return nct2doctors

def get_nct2citations(filename):
    """Return a dictionary: nct2citations."""
    nct2citations = {}
    df = pd.read_csv(filename)
    for i in range(len(df)):
        nct_id, pmid, citation = df['nct_id'][i], str(df['pmid'][i]), str(df['citation'][i])
        if nct_id == 'nan' or citation == 'nan' or pmid == 'nan':
            continue
        if nct_id in nct2citations:
            nct2citations[nct_id].add(citation)
        else:
            nct2citations[nct_id] = {citation}
    return nct2citations

def get_nct2conditions(filename):
    """Return a dictionary: nct2conditions."""
    nct2conditions = {}
    df = pd.read_csv(filename)
    for i in range(len(df)):
        nct_id, term = str(df['nct_id'][i]), str(df['downcase_mesh_term'][i])
        if nct_id == 'nan' or term == 'nan':
            continue
        if nct_id in nct2conditions:
            nct2conditions[nct_id].add(term)
        else:
            nct2conditions[nct_id] = {term}
    return nct2conditions

def get_nct2interventions(filename):
    """Return a dictionary: get_nct2interventions."""
    get_nct2interventions = {}
    df = pd.read_csv(filename)
    for i in range(len(df)):
        nct_id, term = str(df['nct_id'][i]), str(df['downcase_mesh_term'][i])
        if nct_id == 'nan' or term == 'nan':
            continue
        if nct_id in get_nct2interventions:
            get_nct2interventions[nct_id].add(term)
        else:
            get_nct2interventions[nct_id] = {term}
    return get_nct2interventions

def get_all_items(nct2items):
    """Return a list: all_items."""
    all_items = set()
    for nct, items in nct2items.items():
        all_items.update(items)
    return list(all_items)

def get_cit2conditions(all_citations, all_conditions):
    """Return a dictionary: cit2conditions."""
    cit2conditions = {}
    all_conditions = set(all_conditions)
    for cit in all_citations:
        tokens = cit.split()
        cit2conditions[cit] = set()
        for token in tokens:
            if token in all_conditions:
                cit2conditions[cit].add(token)
    return cit2conditions

def get_cit2interventions(all_citations, all_interventions):
    """Return a dictionary: cit2interventions."""
    cit2interventions = {}
    all_interventions = set(all_interventions)
    for cit in all_citations:
        tokens = cit.split()
        cit2interventions[cit] = set()
        for token in tokens:
            if token in all_interventions:
                cit2interventions[cit].add(token)
    return cit2interventions

def build_graph(nct2doctors=False, nct2citations=False, nct2conditions=False, cit2conditions=False, \
                nct2interventions=False, cit2interventions=False):
    """Return a graph with nodes of nct_ids, doctors, citations, conditions and interventions."""
    G = nx.Graph()
    if nct2doctors:
        for nct, doctors in nct2doctors.items():
            G.add_node(nct, category='NCT')
            for doctor in doctors:
                G.add_node(doctor, category='DOC')
                G.add_edge(nct, doctor, weight=1)
    if nct2citations:
        for nct, citations in nct2citations.items():
            G.add_node(nct, category='NCT')
            for citation in citations:
                G.add_node(citation, category='CIT')
                G.add_edge(nct, citation, weight=1)
    if nct2conditions:
        for nct, conditions in nct2conditions.items():
            G.add_node(nct, category='NCT')
            for condition in conditions:
                G.add_node(condition, category='DIS')
                if nct in nct2doctors:
                    for doctor in nct2doctors[nct]:
                        G.add_edge(doctor, condition, weight=1) 
                else:
                    G.add_edge(nct, condition, weight=1)
    if nct2interventions:
        for nct, interventions in nct2interventions.items():
            G.add_node(nct, category='NCT')
            for intervention in interventions:
                G.add_node(intervention, category='DIS')
                G.add_edge(nct, intervention, weight=1)   
    if cit2conditions:
        for cit, conditions in cit2conditions.items():
            G.add_node(cit, category='CIT')
            for condition in conditions:
                G.add_node(condition, category='DIS')
                G.add_edge(cit, condition, weight=1) 
    if cit2interventions:
        for cit, interventions in cit2interventions.items():
            G.add_node(cit, category='CIT')
            for intervention in interventions:
                G.add_node(intervention, category='DIS')
                G.add_edge(cit, intervention, weight=1) 
                        
    return G

def filter_neighbors(graph, node, category):
    """Return a set of neighbors with the certain category of the node in the graph. 
       Category has four values: NCT, DOC, CIT, DIS.
    """
    filtered_neighbors = set()
    for unit in graph[node]:
        if graph.nodes[unit]['category'] == category:
            filtered_neighbors.add(unit)
    return filtered_neighbors

def disjoint(a, b):
    """a, b are sets. Return True if they are disjoint else False."""
    c = set()
    c.update(a)
    c.update(b)
    return True if len(c) - len(a) - len(b) == 0 else False

def find_similar_units(graph, node, n_step, category):
    """Return a dictionary {similar_node: k}. similar_node is node in graph with the certain category that is k steps away from       the origin node.""" 
    if node not in graph:
        return {}
    res, visited = {}, {node: 0}
    queue = [node]
    step = 1
    while step <= n_step:
        new_queue = []
        for unit in queue:
            for item in graph[unit]:
                if item not in visited:
                    visited[item] = step
                    new_queue.append(item)
        queue = new_queue
        step += 1
    
    for unit in visited:
        if graph.nodes[unit]['category'] == category and unit != node:
            res[unit] = visited[unit]
    return res

def read_nct_id_by_year(filename):
    """Return two dictionaries: year2nct_id, nct_id2year. These are trials of FDA new drugs."""
    with open(filename, 'rb') as f:
        year2nct_id = pickle.load(f) 
    nct_id2year = {}
    for year in year2nct_id:
        for nct_id in year2nct_id[year]:
            nct_id2year[nct_id] = year
    return year2nct_id, nct_id2year
