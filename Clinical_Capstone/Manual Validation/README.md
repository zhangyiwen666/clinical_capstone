## Manual Model Validation


1. Pull random pairings under some policy. For example
	- Highest variance across models
	- Top n similarity scores for randomly selected trial
2. Open the urls (https://clinicaltrials.gov/ct2/show/record/**[nct_id]**) and review the Study details page
	- Review conditions, interventions and phase
	- Read the brief summary
3. Record your best estimate of similarity score using the following criteria in your csv
	- 0.0 - Unrelated conditions and interventions
	- 0.2 - Somewhat similar interventions, unrelated conditions 
	- 0.4 - Somewhat similar conditions, unrelated interventions
	- 0.6 - Very similar conditions, somewhat related interventions
	- 0.8 - Very similar conditions and interventions with some differences
	- 1.0 - Conditions and interventions almost identical, same phase
