import pickle
import random
import sys
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <pickle_file_path>")
    sys.exit(1)

data_to_convert=pickle.load(open(sys.argv[1],'rb'))
n_sample=len(data_to_convert[0])
for i in range(n_sample):
    utterance=data_to_convert[0][i]
    pos_response=data_to_convert[1][i]
    print(f"\t[\n\t\t{utterance},\n\t\t{pos_response}],")
