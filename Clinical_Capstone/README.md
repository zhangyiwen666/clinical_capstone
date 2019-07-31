# Goldman Sachs x Columbia Data Science Institute 
# Data Science Capstone (ENGI E4800) - Fall 2018

## `Team`
* Industry Mentor: **Jared Peterson** (Associate, GS)
* DSI Faculty Mentors: 
  1. Professor **Smaranda Muresan** (Data Science Institute, Columbia University)
  2. Professor **Zoran Kostic** (Dept of Electrical Engg, Columbia University)
* DSI Students:
  1. **Aishwarya Srinivasan** (as5431)
  2. **Brian Allen** (ba2542)
  3. **Harsheel Singh Soin** (hss2148)
  4. **Xiangzi Meng** (xmm2103)
  5. **Yiwen Zhang** (yz3310)

## `Introduction`

A unique aspect of the pharmaceutical industry is that a major component of being allowed to sell their product is to successfully conclude an FDA and National Institute of Health supervised clinical trial proving the efficacy and safety of the proposed treatment. Given the importance of the research pipeline to a pharmaceutical firm, it is of interest to determine – based on clinical trials being conducted – which firms are engaged in similar R&D pipelines

Currently, the primary method to determine similarity between clinical trials is using the FDA and NIH-provided keyword categorizations. This existing “keyword” based classification system is based upon the National Library of Medicine’s (NLM) Medical Subject Headings (MeSH)-controlled vocabulary thesaurus. But by just using this appraoch, many more nuanced similarities between two clinical trials may not be assessable. Hence the need for a more advanced technique that uses textual and non-textual information from clinical trial data to assess similarity across them

## `Datasets`

The [AACT](https://www.ctti-clinicaltrials.org/aact-database) (Aggregate Analysis of Clinical Trails) is a publicly available relational database that contains all information (protocol and result data elements) about every study registered in [ClinicalTrials.gov](ClinicalTrials.gov). Content is downloaded from ClinicalTrials.gov daily and loaded into AACT. This study limited to : Investigational New Drug Application (IND) (which means that it is being used to evaluate whether the FDA will approve the drug for public use)

**Useful URLs**:
* [Reporting “Basic Results” in ClinicalTrials.gov](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2821287/)
* [Database Schema](https://aact.ctti-clinicaltrials.org/schema)
* [Helpful Hints](https://aact.ctti-clinicaltrials.org/points_to_consider)
* [Data Dictionary](https://aact.ctti-clinicaltrials.org/data_dictionary)

## `Research Goals`

1. The primary goal of the project is to develop a methodology that will infer a clinical trial’s similarity to another clinical trial using both textual and non-textual information. The similarity resulting from the model should complement, not replace, existing keyword based classification systems
2. The secondary goal of the project is to develop appropriate evaluation criteria that will allow for the comparison and benchmarking of unsupervised clinical trial similarity models

## `Project Phases & Key Dates`

* Phase 1: Background and problem definition (Weeks 1 & 2)
* Phase 2: Data collection, wrangling and cleaning (Weeks 3 & 4)
* Phase 3: Exploratory Data Analysis (Weeks 5,6 & 7)
> First Progress Report Due (Monday, October 22)
* Phase 4: Coding prototypes of algorithms and models (Weeks 8,9 & 10)
* Phase 5: Data visualization and reporting (Weeks 11 & 12)
* Phase 6: Productionizing models or algorithms (Week 13)
> Second Progress Report Due (Monday, November 26)
* Final poster session (Tuesday, December 11)
> Final Report Due (Monday, December 17)

## `Pickle files`
* final\_keyword_model.pkl - https://drive.google.com/file/d/1JVh09YYXiIuj3kXtQgVPta9k0IfFLZHv/view?usp=sharing
* final\_text\_model.pkl  - https://drive.google.com/file/d/1ibr0YjOtMD-_knC03RzQjToCgtFut97E/view?usp=sharing
* final\_graph\_model.pkl - https://drive.google.com/file/d/1KI4jYEpmb4rXvOd58Gz5KzcMP4wQ_aQD/view?usp=sharing
* final\_keyword\_company\_similarity_model.pkl - https://drive.google.com/file/d/10pZngKDQg7A-kUR3li7FCM4ZarjL8kRx/view?usp=sharing
* final\_text\_company\_similarity_model.pkl - https://drive.google.com/file/d/16OPUpQka1sN4oAvNuLh34XiioCeCd-5o/view?usp=sharing
* final\_graph\_company\_similarity_model.pkl - https://drive.google.com/file/d/1k0Un9ChtUAywSsYepzGKvuR_58-CKsHM/view?usp=sharing


## `References`
* [A similarity measurement of clinical trials using SNOMED](https://ieeexplore.ieee.org/document/6867604/)
* [Mapping Similarity Between Clinical Trial Cohorts and US Counties](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5977658/)
* [Clustering clinical trials with similar eligibility criteria features](https://www.ncbi.nlm.nih.gov/pubmed/24496068)
* [Similarities and Difference between Clinical Trials for Foods and Drugs](http://austinpublishinggroup.com/nutrition-food-sciences/fulltext/ajnfs-v5-id1086.php)
