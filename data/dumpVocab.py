import pickle
import random
import sys
if len(sys.argv) < 2:
	print(f"Usage: python {sys.argv[0]} <pickle_file_path>")
	sys.exit(1)

data = pickle.load(open(sys.argv[1],'rb'))
for key, val in data.items():
	print(f"{key}\t{val}")
