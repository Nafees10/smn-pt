# SMN-PT

SMN implementation in PyTorch, updated to work with UDC v2.

[Original Code](https://github.com/chisato-lycoreco/SMN-pytorch)

## Environment

The code has been tested to work in the following environment, using miniconda3:

- gensim==4.3.3
- python==3.8.0
- pytorch==1.7.0
- tensorboardx==2.6.2.2
- tqdm==4.66.5

## Setting Up

1. Clone the repository
1. Place the UDC v2 files (`train.txt` etc) into `/data/udc2`. For example:
	`train.txt` should be located at `/data/udc2/train.txt`
1. Download UDC v1 pkl files from
	[here](https://1drv.ms/u/s!AtcxwlQuQjw1jGn5kPzsH03lnG6U)
1. Extract `word2id.pkl` or similar vocabulary pkl file into `/data/vocab.pkl`
1. Extract `word_embedding.pkl` or `embedding.pkl` to
	`/data/pkl_files/embedding.pkl`
1. `cd` into `/data`
1. Run `python dumpVocab.py vocab.pkl > vocab.orig.txt`
1. Run `python data.py`

## Training & Testing

Run the following to start training and evaluation:

```bash
./run2.sh
```

It will run training with 3 fusion types: `last`, `static`, and `dynamic`. Each
ones output will be saved in `FUSION_TYPE.output` directory.

Recall metrics will be printed at end of training with each fusion type.

## ROUGE

Install the following packages via conda before running ROUGE computation:

- `nltk`
- `conda-forge::rouge-score`
- `absl-py`
- `evaluate`

Run the following to compute ROUGE and print results:

```bash
python compute_rouge.py --num_of_responses 10
```

## Power Usage

To log data about power usage during training, use the `run2-measure.sh` script
instead of `run2.sh`.

This will generate files named `power-log-FUSION_TYPE-PID.csv` and 
`gpu_utilization-FUSION_TYPE-PID.log`. These files can be provided to the
`energy_calc.py` script via CLI parameters, to calculate total energy use.
