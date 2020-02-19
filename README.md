# Learning Directly from Grammar Compressed Text

## Dependencies
### Compression Software
#### LZD

Our codes use slightly modified version of publicly available implementation of LZD algorithm, **LZD**, which can be found at https://github.com/aurtg/lzd.

(You can find the original version of **LZD** in https://github.com/kg86/lzd.)

Please follow instructions in Compile section of  `README.md` to compile the software.

#### Re-Pair

Our codes use slightly modified version of publicly available implementation of Re-Pair algorithm, **Re-Pair-VF**, which can be found at https://github.com/aurtg/Re-Pair-VF.

(You can find the original version of **Re-Pair-VF** in https://github.com/syoshid/Re-Pair-VF.)

Please run a following command in the root directory of the software to compile it.

```
make txt2cfg
```

### Python Libraries

Our codes are developed on Python 3.6.8 with following libraries.

* numpy (1.17.4)
* scikit-learn (0.19.1)
* pytorch (1.1.0)
  - https://pytorch.org/
* torch-scatter (1.3.1)
  - https://github.com/rusty1s/pytorch_scatter
* pybind11
  - https://github.com/pybind/pybind11
  - Please install it through `pip`.

## How to run

Unless otherwise noted, execute following commands or run following scripts at the root directory of this repository.

### Preparation

#### 1. Build C++ code

```
# Compile c++ module using pybind11
bash src/build.sh
```

#### 2. Edit configurations

Specify position of executable files of two compression softwares, **LZD** and **Re-Pair-VF**.

**LZD**: Specify position of `out/lzd` at Line 8 of `src/lzd.py`

**Re-Pair-VF**: Specify position of `txt2cfg` at Line 10 of `src/repair.py`

#### 3. Download & Preprocess Datasets

Please follow instructions in `data/dna/README.md` and `data/text/README.md`.

**Note: Preprocessing whole datasets may take long time (a few days).**

### Run experiments

#### Commands

Run following python scripts to train and evaluate models.
See detail descriptions of options by running scripts with `--help` option.
Result of an experiment will be logged in `logs/<experiment suffix>.log`.

**Bi-LSTM (baseline)**: `src/baselines/naive_lstm.py`

**Proposed Method**: `src/lzd_lstm.py`

* You can choose a compression algorithm used in experiments by setting `--alg` option to `lzd` (LZD) or `repair` (Re-Pair).
* You can choose a composer module used in experiments by setting `--composer` to `fully-connected` (MLP) or `dual-gru` (Dual-GRU).

For dataset specific options (`--n-char` and `--n-cls`), see following *Dataset Specific Parameters* subsection.

#### Dataset specific parameters

| Dataset | n-char | n-cls|
| --- | --- | --- |
| DNA  | 5  | 2  |
| Text (sogou_news_csv)   |  72 | 5  |
| Text (yelp_review_full_csv)   | 72  | 5  |
| Text (yelp_review_polarity_csv)   | 72  | 2  |

* `n-char` specifies the number of character types in each dataset.
  - It includes one dummy character type for padding.
* `n-cls` specifies the number of classification categories in each dataset.

#### Example

The following example command will run experiment of proposed method on Yelp Review Full dataset of text classification task using GPU 0 with the LZD compression algorithm.

```
python -u src/lzd_lstm.py --dataset data/text/yelp_review_full_csv --alg lzd --mode test --n-cls 5 --n-char 72 --bs 126 --lr-decay-factor 2 --warmup-step 1000 --lr 1e-3 --lr-decay-freq 10 --decay 0.0 --h-dim 200 --n-lstm-layers 1 --dropout 0.5 --epoch 50 --n-epoch-data 100000 --grad-accum-step 3 --gpu 0
```

## License

Codes in this repository except a dataset file, data/dna/dataset.tar.gz, are licensed under the GNU General Public License (v3) (see ./LICENSE).
For non-academic commercial use, please [contact us](mailto:kosuke_a@nec.com) for commercial licensing.
