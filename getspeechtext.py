from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models.widgets import TextInput, Button, Paragraph
import pickle

raw_speeches = pickle.load(open("raw_speeches.pickle", "rb"))
speechid_to_speaker = pickle.load(open("speechid_to_speaker.pickle", "rb"))

button = Button(label = "Get Speech")
input = TextInput(value = "Speechid")
output = Paragraph()
output2 = Paragraph()
output3 = Paragraph()

def update():
	output.text = "" + speechid_to_speaker[input.value]
	output2.text = "\r\n"
	output3.text = raw_speeches[input.value]
button.on_click(update)
layout = column(input, button, output, output2, output3)
curdoc().add_root(layout)