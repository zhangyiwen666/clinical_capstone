import pickle
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.plotting import figure
from bokeh.models.widgets import Select, Slider, PreText
from bokeh.models import ColumnDataSource

#####################
### IMPORT MODELS ###
#####################

keyword = pickle.load(open("Output/final_keyword_company_similarity_model.pkl", "rb"))
text = pickle.load(open("Output/final_text_company_similarity_model.pkl", "rb"))
graph = pickle.load(open("Output/final_graph_company_similarity_model.pkl", "rb"))


###########################
### IMPORT DICTIONARIES ###
###########################

sponsor_2_nct_ids_by_year = pickle.load(open("Data/sponsor_2_nct_ids_by_year.pkl", "rb"))
ids_by_year = pickle.load(open("Data/ids_by_year_fda_reg_with_pos.pkl", "rb"))

year_list = list(range(2014,2019))

sponsor_by_year_with_pos = {year:{elem:i for i,elem in enumerate(sorted(list(sponsor_2_nct_ids_by_year[year].keys())))} for year in year_list}
sponsor_by_year_with_pos_rev = {year:{i:elem for i,elem in enumerate(sorted(list(sponsor_2_nct_ids_by_year[year].keys())))} for year in year_list}


companies = set()
for year in year_list:
    for sponsor in sponsor_by_year_with_pos[year].keys():
        companies.add(sponsor)

companies = sorted(list(companies))



#############
### BOKEH ###
#############

# Define toggles and widgets
year_select = Select(title="Year:", value='2018', options=[str(year) for year in year_list])
company_select = Select(title="Company:", value='Bayer', options=companies)
keyword_slider = Slider(start=0, end=1, value=1, step=.01, title="Keyword relative weight")
text_slider = Slider(start=0, end=1, value=1, step=.01, title="Text relative weight")
graph_slider = Slider(start=0, end=1, value=1, step=.01, title="Graph relative weight")
num_companies_select = Slider(start=1, end=100, value=20, step=1, title="Number of comparisons to display")
beta_text = PreText(text='', width=200)
comparison_text = PreText(text='', width=750)



def get_betas():
    """
    Using the relative weights, generate betas that sum to 1
    """
    beta_total = keyword_slider.value + text_slider.value + graph_slider.value
    if beta_total==0:
        return 1/3, 1/3, 1/3
    beta1 = keyword_slider.value / beta_total
    beta2 = text_slider.value / beta_total
    beta3 = graph_slider.value / beta_total

    return beta1, beta2, beta3

b1, b2, b3 = get_betas()


def get_similarities(b1, b2, b3, year, company):
    """
    Weight the scores and return a sorted dataframe of company and similarity
    """
    company_pos = sponsor_by_year_with_pos[year][company]

    values = keyword[year][company_pos]*b1 + text[year][company_pos]*b2 + graph[year][company_pos]*b3
    sponsor_list = sponsor_by_year_with_pos_rev[year].values()

    company_scores = pd.DataFrame(sorted(list(zip(sponsor_list, values)), key=lambda x: x[1], reverse=True))
    company_scores.columns = ['company', 'similarity']
    return company_scores

# Store data in a column data source
source = ColumnDataSource(data=dict(company=[], x=[]))

TOOLTIPS=[
    ("Company", "@company"),
    ("Similarity", "@x")
]

# Generate rug plot p
p = figure(plot_height=300, plot_width=600, title="", toolbar_location=None, tooltips=TOOLTIPS, x_axis_label='Similarity Score')
p.circle(x="x", y=0, source=source, size=20, line_color=None)


def update():
    """
    Update all values when something changes
    """
    b1, b2, b3 = get_betas()
    df = get_similarities(b1, b2, b3, int(year_select.value), company_select.value)
    source.data = dict(
        company=df.iloc[1:num_companies_select.value+1]['company'],
        x=df.iloc[1:num_companies_select.value+1]['similarity'],
    )
    beta_text.text = "\nkeyword beta: {:.2f}\n\n\ntext beta: {:.2f}\n\n\ngraph beta: {:.2f}".format(b1, b2, b3)
    comparison_text.text = str(df.iloc[1:num_companies_select.value+1])


# Initialize update
update()


# Store controls that when changed trigger and update
controls = [year_select, company_select, num_companies_select, keyword_slider, text_slider, graph_slider]

for control in controls:
    control.on_change('value', lambda attr, old, new: update())


# Format output
sliders = column(keyword_slider,
                 text_slider,
                 graph_slider)

sliders_text = row(beta_text, sliders)

widgets = column(year_select,
                 company_select,
                 num_companies_select,
                 sliders_text,
                 width=300)
comparisons = column(p, comparison_text, width=750)
layout = row(comparisons, widgets)

curdoc().add_root(layout)
curdoc().title = "Company comparison tool"