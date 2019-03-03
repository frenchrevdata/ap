from google_drive_downloader import GoogleDriveDownloader as gdd
from bokeh.io import curdoc
from bokeh.models.widgets import TextInput, Button, Paragraph, Div
import pickle
import pandas as pd
import os
from bokeh.layouts import *
from bokeh.models import ColumnDataSource
from bokeh.models import *
from bokeh.plotting import *
from bokeh.embed import file_html
from bokeh.resources import CDN
import itertools

curr_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(curr_dir, 'data')
# check to see if data directory exists, if not make it
if not os.path.isdir(data_dir):
	os.mkdir(data_dir)


def load_pickle_file(gdriveurl, filename):
	pathname = os.path.join(data_dir, filename)
	if not os.path.isfile(pathname):
		gdd.download_file_from_google_drive(file_id=gdriveurl, dest_path=pathname)
	with open(pathname, 'rb') as f:
		return pickle.load(f)

# raw_speeches = load_pickle_file('1ghlAIXa9pBq1Qvc2JLfZMlb7uD2v6jCB', 'raw_speeches.pickle')
speechid_to_speaker = load_pickle_file('1j2GGzjTrrzCvoAMpa08mtQnoTNbt4Kbe', 'speechid_to_speaker.pickle')

chronology_date = load_pickle_file('1JE6K0mj0ZINb0loDfgx1FSlqVlSDnm-f', 'chronology_date.pickle')
chronology_date["Bigram"] = chronology_date["Bigram"].astype(str)
chronology_date["Date"] = pd.to_datetime(chronology_date.Date, format = "%Y-%m-%d")

pathname = os.path.join(data_dir, 'speeches.zip')
if not os.path.isfile(pathname):
	gdd.download_file_from_google_drive(file_id='1s4r7LiKU95QQ0-SD9IevIxPI_An-2yIi', dest_path = pathname)
	os.system('unzip ' + pathname)

buttonspeech = Button(label = "Get Speech")
input = TextInput(value = "Speechid")
output = Paragraph()
output3 = Paragraph()

def get_data(bigram):
	group = chronology_date.loc[chronology_date["Bigram"] == bigram]
	num_per_date = group.groupby(["Date"]).agg({"Num occurrences": "sum"})
	xs = num_per_date.index.tolist()
	ys = num_per_date["Num occurrences"].tolist()
	return dict(x=xs,y=ys)

# inputp = TextInput(value = "(u'convention', u'renvoie')", width = 500)
# buttonp = Button(label="Plot")

hover = HoverTool(
	tooltips = [
	("date", "@x{%F}"),
	("y", "$y"),
	],
	formatters = {"x":"datetime"},
	mode = "mouse")

inputp = TextInput(value = "(u'convention', u'renvoie')", width = 500)
buttonp = Button(label="Plot", width = 500)

p = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcep = ColumnDataSource(data = get_data("(u'convention', u'renvoie')"))
p.line(x='x', y='y', line_width = 2, source = sourcep)

inputq = TextInput(value = "(u'convention', u'renvoie')", width = 500)
buttonq = Button(label="Plot", width = 500)

q = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourceq = ColumnDataSource(data = get_data("(u'convention', u'renvoie')"))
q.line(x='x', y='y', line_width = 2, line_color = "green", source = sourceq)

inputr = TextInput(value = "(u'convention', u'renvoie')", width = 500)
buttonr = Button(label="Plot", width = 500)

r = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcer = ColumnDataSource(data = get_data("(u'convention', u'renvoie')"))
r.line(x='x', y='y', line_width = 2, line_color = "red", source = sourcer)

inputs = TextInput(value = "(u'convention', u'renvoie')", width = 500)
buttons = Button(label="Plot", width = 500)

s = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sources = ColumnDataSource(data = get_data("(u'convention', u'renvoie')"))
s.line(x='x', y='y', line_width = 2, line_color = "blueviolet", source = sources)

inputt = TextInput(value = "(u'convention', u'renvoie')", width = 500)
buttont = Button(label="Plot", width = 500)

t = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcet = ColumnDataSource(data = get_data("(u'convention', u'renvoie')"))
t.line(x='x', y='y', line_width = 2, line_color = "orange", source = sourcet)

inputw = TextInput(value = "(u'convention', u'renvoie')", width = 500)
buttonw = Button(label="Plot", width = 500)

w = figure(x_axis_type = "datetime", plot_width = 500, plot_height = 500, tools = [hover], title = "Bigram over time")
sourcew = ColumnDataSource(data = get_data("(u'convention', u'renvoie')"))
w.line(x='x', y='y', line_width = 2, line_color = "teal", source = sourcew)


def update():
	filename = input.value + ".pickle"
	pathname = os.path.join(curr_dir, 'Speeches/' + filename)
	# pathname = os.path.join(data_dir, filename)
	if os.path.isfile(pathname):
		output.text = "" + speechid_to_speaker[input.value]
		with open(pathname, 'rb') as f:
			speech = pickle.load(f)
		output3.text = "" + speech

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

buttonspeech.on_click(update)
buttonp.on_click(update)
buttonq.on_click(update)
buttonr.on_click(update)
buttons.on_click(update)
buttont.on_click(update)
buttonw.on_click(update)


# layout = column(input, button, p)
l = layout([column(input, buttonspeech, output),
	column([output3], sizing_mode = "scale_height"),
	[Spacer(height=150), Spacer(height=150), Spacer(height = 150)],
	[inputp, inputq, inputr], 
	[buttonp, buttonq, buttonr], 
	[p, q, r], 
	[inputs, inputt, inputw], 
	[buttons, buttont, buttonw], 
	[s, t, w]])

curdoc().add_root(l)