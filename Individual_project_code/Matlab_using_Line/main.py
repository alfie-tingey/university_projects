from queueing_rnn import RNNCell, RNNCellTraining
import sys


def help():
	print('\n'.join([
		"Usage: ",
		f"python3 {sys.argv[0]} <matdir> <what> <output_file>",
		"where",
		"matdir: directory that contains learning traces in .mat format",
		"what: which setting to use (synthetic, real)",
		"output_file: where to save the learnt parameters"]))


#if len(sys.argv) != 4:
	#help()

matdir = '/Users/alfredtingey/RNN_queueing/learning_traces/synthetic/net_5_generated'
what = 'synthetic'
output_file = '/Users/alfredtingey/RNN_queueing/models_learnt/model_5_generated.txt'

td = RNNCellTraining(matdir, lambda init_s: RNNCell(init_s))

if what == 'synthetic':
	td.load_file()
	td.makeNN(lr=0.05)
	td.learn()
	td.saveResults(output_file)
else:
	td.load_file()
	td.makeNN(lr=0.01)
	td.learn()
	td.saveResults(output_file)
