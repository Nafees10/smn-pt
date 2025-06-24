import sys

def chunker(file_path : str, chunk_size : int):
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line.rstrip('\n'))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def precision_calculate(file_path):
    predicted_positive = 0
    positive_correct = 0
    for lines in chunker(file_path, 10):
        splited_lines = [line.split("\t") for line in lines]
        scores = [float(split[0]) for split in splited_lines]
        labels = [int(split[1]) for split in splited_lines]
        for score, label in zip(scores, labels):
            if score > 0.5:
                predicted_positive += 1
                if label > 0:
                    positive_correct += 1
    return float(positive_correct) / (predicted_positive + 1e-12)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} path/to/out.txt")
    print(precision_calculate(sys.argv[1]))
