from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import TextInput, Button, Paragraph
from bokeh.models import *
from bokeh.plotting import *
from bokeh.embed import file_html
from bokeh.resources import CDN
import itertools
import pickle
import pandas as pd 


chronology_date = pickle.load(open("chronology_date.pickle", "rb"))
chronology_date["Bigram"] = chronology_date["Bigram"].astype(str)
chronology_date["Date"] = pd.to_datetime(chronology_date.Date, format = "%Y-%m-%d")


def get_data(bigram):
	group = chronology_date.loc[chronology_date["Bigram"] == bigram]
	num_per_date = group.groupby(["Date"]).agg({"Num occurrences": "sum"})
	xs = num_per_date.index.tolist()
	ys = num_per_date["Num occurrences"].tolist()
	return dict(x=xs,y=ys)

input = TextInput(value = "('convention', 'renvoie')", width = 500)
button = Button(label="Plot")

hover = HoverTool(
	tooltips = [
	("date", "@x{%F}"),
	("y", "$y"),
	],
	formatters = {"x":"datetime"},
	mode = "mouse")

p = figure(x_axis_type = "datetime", plot_width = 1200, plot_height = 800, tools = [hover], title = "Bigram over time")
source = ColumnDataSource(data = get_data("('convention', 'renvoie')"))
p.line(x='x', y='y', line_width = 2, source = source)



def update():
	new_data = get_data(input.value)
	source.data = new_data

button.on_click(update)
layout = column(input, button, p)

curdoc().add_root(layout)
# curdoc().add_root(p)
output_file("bigramplots.html", mode='inline')
save(p)

