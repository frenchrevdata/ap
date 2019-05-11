from google_drive_downloader import GoogleDriveDownloader as gdd
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.widgets import TextInput, Button, Paragraph
import pickle
import pandas
import os


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
pathname = os.path.join(data_dir, 'speeches.zip')
if not os.path.isfile(pathname):
	gdd.download_file_from_google_drive(file_id='1s4r7LiKU95QQ0-SD9IevIxPI_An-2yIi', dest_path = pathname)
	os.system('unzip ' + pathname)

button = Button(label = "Get Speech")
input = TextInput(value = "Speechid")
output = Paragraph()
output2 = Paragraph()
output3 = Paragraph()

def update():
	output.text = "" + speechid_to_speaker[input.value]
	output2.text = "\r\n"
	filename = input.value + ".pickle"
	pathname = os.path.join(curr_dir, 'Speeches/' + filename)
	# pathname = os.path.join(data_dir, filename)
	with open(pathname, 'rb') as f:
		speech = pickle.load(f)
	output3.text = "" + speech
	# output3.text = raw_speeches[input.value]
button.on_click(update)
layout = column(input, button, output, output2, output3)
curdoc().add_root(layout)