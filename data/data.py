import pickle
import os
from gensim.models import Word2Vec
import numpy as np

class SentenceIterator:
    def __init__(self, files, respPath = None):
        self.files = files
        self.respPath = respPath

    def __iter__(self):
        for file in self.files:
            with open(file, encoding='utf-8') as f:
                for line in f:
                    dialogue = line.strip().split('\t')[1]
                    yield dialogue.split()
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
                for word in line.strip().split("\t")[1]:
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
    for _, id in enumerate(ret):
        zeroCount += id == 0
    return ret

def responsesLoad(path, vocab):
    responses = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                id, utt = line.strip().split('\t', 1)
                id = int(id)
                responses[id] = tokenize(utt, vocab)
            except:
                pass
    print(f"responses length: {len(responses)}")
    return responses

def embeddingsBuild(vocab, files, respPath=None):
    sentences = SentenceIterator(files, respPath=respPath)
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5,
                     workers=4)

    matrix = np.zeros((len(vocab) + 1, 200))
    for word, idx in vocab.items():
        if word in model.wv:
            matrix[idx] = model.wv[word]
    print(f"matrix length: {len(matrix)}")
    return matrix

def embeddingsLoad(vocab, embPath):
    matrix = np.zeros((len(vocab) + 1, 400))
    with open(embPath) as f:
        for line in f:
            splits = line.split()
            word = " ".join(splits[0:len(splits) - 400])
            id = vocab.get(word, -1)
            if id == 0:
                continue
            matrix[id] = np.array([np.float16(v)
                                   for v in splits[len(splits)-400:]])
    return matrix

def dataProcess(path, vocab, responses, enforce=False):
    uttList = []
    resList = []
    labList = []
    labCount = {0:0, 1:0}
    skipped = 0
    with open(path, 'r') as f:
        for line in f:
            _, utt, valIds, invIds = \
                line.strip().split('\t')
            utt = utt.replace("__eot__", "")
            uttToks = [] + [
                tokenize(turn, vocab) for turn in utt.split("__eou__")
                    if len(turn.strip()) > 0
            ]
            valRes = [
                responses[int(id)] for id in valIds.split('|')
                    if id != "NA"
            ]
            invRes = [
                responses[int(id)] for id in invIds.split('|')
                    if id != "NA"
            ]
            if enforce and (len(invRes) != 9 or len(valRes) != 1):
                skipped += 1
                continue
            for res in valRes:
                uttList.append(uttToks)
                resList.append(res)
                labList.append(1)
                labCount[1] += 1
            for res in invRes:
                uttList.append(uttToks)
                resList.append(res)
                labList.append(0)
                labCount[0] += 1
    print(f"{path}: uttList length: {len(uttList)}")
    print(f"{path}: labels: {labCount}")
    if enforce:
        print(f"{path}: skipped due to enforce: {skipped}")
    return uttList, resList, labList

def filesBuild(vocabPath, responsesPath, trainPath, testPath,\
        validPath, embPath, outDir):
    os.makedirs(outDir, exist_ok=True)
    print("Getting vocab...")
    vocab = vocabLoad(vocabPath)
    """
    vocab = vocabGen([trainPath, testPath, validPath],
                     vocabPath=vocabPath,
                     respPath=responsesPath)
    """
    print("\tdone")
    responses = responsesLoad(responsesPath, vocab)

    print("processing train data...")
    trainData = dataProcess(trainPath, vocab, responses)
    with open(os.path.join(outDir, 'utterances.pkl'), 'wb') as f:
        pickle.dump(trainData, f)
    print("\tdone")

    print("processing eval data...")
    evalData = dataProcess(validPath, vocab, responses, True)
    with open(os.path.join(outDir, 'Evaluate.pkl'), 'wb') as f:
        pickle.dump(evalData, f)
    print("\tdone")

    """
    print("training embeddings...")
    embeddingMatrix = embeddingsBuild(vocab,
                                      [trainPath, validPath, testPath],
                                      respPath=responsesPath)
    embeddingMatrix = embeddingsLoad(vocab, embPath)
    with open(os.path.join(outDir, 'embedding.pkl'), 'wb') as f:
        pickle.dump(embeddingMatrix, f)
    """
    print("\tdone")

    print("Generated required .pkl files in:", outDir)

vocabPath = 'vocab.orig.txt'
responsesPath = 'udc2/responses.txt'
trainPath = 'udc2/train.txt'
testPath = 'udc2/test.txt'
embPath = 'udc2/embedding.txt'
#validPath = 'udc2/valid.txt'
validPath = 'udc2/test.txt' # TODO
outDir = './pkl_files'

filesBuild(vocabPath, responsesPath, trainPath, testPath,
           validPath, embPath, outDir)
print(f"zeroCount = {zeroCount}")
