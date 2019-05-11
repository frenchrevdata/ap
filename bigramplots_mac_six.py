from bokeh.io import curdoc
from bokeh.layouts import *
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import TextInput, Button, Paragraph, Div
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

# inputp = TextInput(value = "('convention', 'renvoie')", width = 500)
# buttonp = Button(label="Plot")

hover = HoverTool(
	tooltips = [
	("date", "@x{%F}"),
	("y", "$y"),
	],
	formatters = {"x":"datetime"},
	mode = "mouse")

inputp = TextInput(value = "('convention', 'renvoie')", width = 500)
buttonp = Button(label="Plot", width = 500)

p = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcep = ColumnDataSource(data = get_data("('convention', 'renvoie')"))
p.line(x='x', y='y', line_width = 2, source = sourcep)

inputq = TextInput(value = "('convention', 'renvoie')", width = 500)
buttonq = Button(label="Plot", width = 500)

q = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourceq = ColumnDataSource(data = get_data("('convention', 'renvoie')"))
q.line(x='x', y='y', line_width = 2, line_color = "green", source = sourceq)

inputr = TextInput(value = "('convention', 'renvoie')", width = 500)
buttonr = Button(label="Plot", width = 500)

r = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcer = ColumnDataSource(data = get_data("('convention', 'renvoie')"))
r.line(x='x', y='y', line_width = 2, line_color = "red", source = sourcer)

inputs = TextInput(value = "('convention', 'renvoie')", width = 500)
buttons = Button(label="Plot", width = 500)

s = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sources = ColumnDataSource(data = get_data("('convention', 'renvoie')"))
s.line(x='x', y='y', line_width = 2, line_color = "blueviolet", source = sources)

inputt = TextInput(value = "('convention', 'renvoie')", width = 500)
buttont = Button(label="Plot", width = 500)

t = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcet = ColumnDataSource(data = get_data("('convention', 'renvoie')"))
t.line(x='x', y='y', line_width = 2, line_color = "orange", source = sourcet)

inputw = TextInput(value = "('convention', 'renvoie')", width = 500)
buttonw = Button(label="Plot", width = 500)

w = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcew = ColumnDataSource(data = get_data("('convention', 'renvoie')"))
w.line(x='x', y='y', line_width = 2, line_color = "teal", source = sourcew)


def update():
	new_datap = get_data(inputp.value)
	sourcep.data = new_datap

	new_dataq = get_data(inputq.value)
	sourceq.data = new_dataq

	new_datar = get_data(inputr.value)
	sourcer.data = new_datar

	new_datas = get_data(inputs.value)
	sources.data = new_datas

	new_datat = get_data(inputt.value)
	sourcet.data = new_datat

	new_dataw = get_data(inputw.value)
	sourcew.data = new_dataw

buttonp.on_click(update)
buttonq.on_click(update)
buttonr.on_click(update)
buttons.on_click(update)
buttont.on_click(update)
buttonw.on_click(update)

# layout = column(input, button, p)
l = layout([[inputp, inputq, inputr], 
	[buttonp, buttonq, buttonr], 
	[p, q, r], 
	[inputs, inputt, inputw], 
	[buttons, buttont, buttonw], 
	[s, t, w]], sizing_mode = "fixed")

curdoc().add_root(l)
# curdoc().add_root(p)
# output_file("bigramplots.html", mode='inline')
# save(p)

