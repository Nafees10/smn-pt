import pickle
import os
from gensim.models import Word2Vec
import numpy as np
from sys import stderr

class SentenceIterator:
    def __init__(self, files, respPath = None):
        self.files = files
        self.respPath = respPath

    def __iter__(self):
        for file in self.files:
            with open(file, encoding='utf-8') as f:
                for line in f:
                    dialogue = line.strip().split('\t')[1:]
                    for turn in dialogue:
                        yield turn.split()
        if self.respPath == None:
            return
        with open(self.respPath, encoding='utf-8') as f:
            for line in f:
                yield line.split("\t")[1].split()

def vocabLoad(path):
    ret = {}
    with open(path, 'r') as f:
        for line in f:
            word, id = line.strip().split('\t')
            ret[word] = int(id)
    print(f"vocab length: {len(ret)}")
    return ret

def vocabGen(paths, respPath = None, vocabPath = None):
    ret = {"unknown": 0}
    if vocabPath != None:
        ret = vocabLoad(vocabPath)
    for path in paths:
        with open(path) as f:
            for line in f:
                for utt in line.strip().split("\t")[1:]:
                    for word in utt.split():
                        if word in ret:
                            continue
                        ret[word] = len(ret)
    if respPath == None:
        print(f"vocab length: {len(ret)}")
        return ret
    with open(respPath) as f:
        for line in f:
            for word in line.split("\t")[1].split():
                if word in ret:
                    continue
                ret[word] = len(ret)
    print(f"vocab length: {len(ret)}")
    return ret;

zeroCount = 0

def tokenize(utt, vocab):
    global zeroCount
    ret = [vocab.get(tok, 0) for tok in utt.split()]
    for i, id in enumerate(ret):
        zeroCount += id == 0
        print(f"zero: {utt.split()[i]}", file=stderr)
    return ret

def embeddingsBuild(vocab, files, respPath = None):
    sentences = SentenceIterator(files, respPath)
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5,
                     workers=4, epochs=20)
    matrix = np.zeros((len(vocab) + 1, 200))
    for word, idx in vocab.items():
        if word in model.wv:
            matrix[idx] = model.wv[word]
    print(f"matrix length: {len(matrix)}")
    return matrix

def dataProcess(path, vocab):
    uttList = []
    resList = []
    labList = []
    labCount = {0:0, 1:0}
    with open(path, 'r') as f:
        for line in f:
            splits = line.strip().split("\t")
            label = int(splits[0])
            utts = splits[1:-1]
            res = splits[-1]
            uttToks = [] + [tokenize(turn, vocab) for turn in utts]
            resToks = tokenize(res, vocab)
            uttList.append(uttToks)
            resList.append(resToks)
            labList.append(label)
            labCount[label] += 1
    print(f"{path}: uttList length: {len(uttList)}")
    print(f"{path}: labels: {labCount}")
    return uttList, resList, labList

def filesBuild(vocabPath, trainPath, testPath,\
        validPath, outDir):
    os.makedirs(outDir, exist_ok=True)
    print("Getting vocab...")
    #vocab = vocabLoad(vocabPath)
    vocab = vocabGen([trainPath, testPath, validPath], vocabPath=vocabPath)
    print("\tdone")

    print("processing train data...")
    trainData = dataProcess(trainPath, vocab)
    with open(os.path.join(outDir, 'utterances.pkl'), 'wb') as f:
        pickle.dump(trainData, f)
    print("\tdone")

    """
    print("training embeddings...")
    embeddingMatrix = embeddingsBuild(vocab, [trainPath, validPath, testPath])
    with open(os.path.join(outDir, 'embedding.pkl'), 'wb') as f:
        pickle.dump(embeddingMatrix, f)
    print("\tdone")
    """

    print("processing eval data...")
    evalData = dataProcess(validPath, vocab)
    with open(os.path.join(outDir, 'Evaluate.pkl'), 'wb') as f:
        pickle.dump(evalData, f)
    print("\tdone")

    print("Generated required .pkl files in:", outDir)

vocabPath = '/home/nafees/ddata/ubuntu/vocab.txt'
trainPath = '/home/nafees/ddata/ubuntu/train.txt'
testPath = '/home/nafees/ddata/ubuntu/test.txt'
validPath = '/home/nafees/ddata/ubuntu/valid.txt'
outDir = './pkl_files'

filesBuild(vocabPath, trainPath, testPath, validPath, outDir)
print(f"zeroCount = {zeroCount}")
