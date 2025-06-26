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

Begin by cloning the repository.

### Data Preprocessing

Before training can begin, the dataset needs to be transformed into some `.pkl`
files. For this, you need the UDC v2's `.txt` files, and 2 files from the
preprocessed `.pkl` files provided by original SMN model's authors:
`embedding.pkl` and `worddict.pkl`.

The `data/dumpVocab.py` script is used to dump vocabulary from UDC v1's
`worddict.pkl` file into text format, resembling UDC v2's `vocab.txt`. This
file will help make sense of the existing `embedding.pkl` we will use.

The `data/data.py` script is used to convert UDC v2 format dataset into
`.pkl` files that can be used by SMN.

#### Extracting UDC v1 files:

1. Download UDC v1 pkl files from
	[here](https://1drv.ms/u/s!AtcxwlQuQjw1jGn5kPzsH03lnG6U)
1. Extract `embedding.pkl` to `smn-pt/data/pkl_files/embedding.pkl`
1. Extract `worddict.pkl` to `smn-pt/data/worddict.pkl`

Dump vocab from `worddict.pkl` to UDC v2 format:
```bash
cd data
python dumpVocab.py worddict.py > vocab.orig.txt
```

#### Pre-processing UDC v2 files:

1. Place the UDC v2 files (`train.txt` etc) into `smn-pt/data/udc2`.
	For example: `train.txt` should be located at `smn-pt/data/udc2/train.txt`
1. Run `cd data; python data.py`
1. `pkl` files will be created in the `smn-pt/data/pkl_files` directory.

## Training & Testing

Run the following to start training and evaluation:

```bash
./run2.sh
```

It will run training with 3 fusion types: `last`, `static`, and `dynamic`. Each
ones output will be saved in `FUSION_TYPE.output` directory.

Recall metrics will be printed at end of training with each fusion type.

The code does not have an option to load an existing trained model, and run
evaluation. Instead, evaluation is ran at end of training, and the model is
saved.

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

## Precision

To calculate precision, use the `precision.py` script after training &
evaluation:

```bash
python precision.py last.output/out.txt
python precision.py static.output/out.txt
python precision.py dynamic.output/out.txt
```
