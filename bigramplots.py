from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import TextInput, Button, Paragraph
from bokeh.models import *
from bokeh.plotting import *
from bokeh.embed import file_html
from bohek.resources import CDN
import itertools
import pickle
import pandas as pd 

chronology_date = pickle.load(open("chronology_date.pickle", "rb"))
chronology_date["Bigram"] = chronology_date["Bigram"].astype(str)
chronology_date["Date"] = pd.to_datetime(chronology_date.Date, format = "%Y-%m-%d")

output_file("bigramplots.html")
# bigram = "(u'convention', u'renvoie')"
# # print chronology_date["Bigram"]
# # num_bigram_date = pickle.load(open("num_per_bigram_date.pickle","rb"))
# # print num_bigram_date.columns
# # num_bigram_date = pd.to_datetime(num_bigram_date["Date"], format = "%Y-%m-%d")
# group = chronology_date.loc[chronology_date["Bigram"] == "(u'convention', u'renvoie')"]
# # print group
# num_per_date = group.groupby(["Date"]).agg({"Num occurrences": "sum"})
# print num_per_date.index
# # print group
# # grouped_by_bigram = pickle.load(open("bybigram.pickle", "rb"))
# # grouped_by_bigram = pd.to_datetime(grouped_by_bigram["Date"], format = "%Y-%m-%d")
# # group = grouped_by_bigram.get_group("(u'convention', u'renvoie')")
# xs = num_per_date.index.tolist()
# print xs
# ys = num_per_date["Num occurrences"].tolist()
# print ys

# hover = HoverTool(
# 	tooltips = [
# 	("date", "@x{%F}"),
# 	("y", "$y"),
# 	],
# 	formatters = {"x":"datetime"},
# 	mode = "mouse")


# plot_title = bigram + " over time"
# p = figure(x_axis_type = "datetime", plot_width = 800, plot_height = 800, tools = [hover], title = plot_title)
# p.xaxis.axis_label = "Date"
# p.yaxis.axis_label = "Number of Occurrences"
# # p.circle(xs, ys, size =15, line_color="navy", fill_alpha = 0.5)
# p.line(xs, ys, line_width = 2)
# show(p)

def get_data(bigram):
	group = chronology_date.loc[chronology_date["Bigram"] == bigram]
	num_per_date = group.groupby(["Date"]).agg({"Num occurrences": "sum"})
	xs = num_per_date.index.tolist()
	ys = num_per_date["Num occurrences"].tolist()
	return dict(x=xs,y=ys)

# def modify_plot(doc):
input = TextInput(value = "(u'convention', u'renvoie')", width = 500)
button = Button(label="Plot")

hover = HoverTool(
	tooltips = [
	("date", "@x{%F}"),
	("y", "$y"),
	],
	formatters = {"x":"datetime"},
	mode = "mouse")

p = figure(x_axis_type = "datetime", plot_width = 800, plot_height = 800, tools = [hover], title = "Bigram over time")
# p.xaxis.axis_label = "Date"
# p.yaxis.axis_label = "Number of Occurrences"
# p.circle(xs, ys, size =15, line_color="navy", fill_alpha = 0.5)
source = ColumnDataSource(data = get_data("(u'convention', u'renvoie')"))
# p.line(xs, ys, line_width = 2)
p.line(x='x', y='y', line_width = 2, source = source)

# callback = CustomJS(args=dict(source=source), code="""
# 	var data = source.get('data');
# 	var f = cb_obj.get('value')
# 	print f
# 	x = get_data(f).keys()
# 	y = get_data(f).values()
# 	source.trigger('change');
# 	""")

def update():
	new_data = get_data(input.value)
	source.data = new_data
# input = TextInput(value = "(u'convention', u'renvoie')", width = 500, callback = callback)
button.on_click(update)
layout = column(input, button)
# curdoc().add_periodic_callback(update, 150)
curdoc().add_root(layout)
curdoc().add_root(p)

# show(modify_plot)