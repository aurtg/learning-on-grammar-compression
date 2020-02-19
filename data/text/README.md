1. Download compressed dataset '*.tar.gz' into this directory from [Here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)

2. Extract dataset:
`for file in $(ls *.tar.gz); do tar -zxvf ${file}; done`

3. Preprocess dataset: `python data/text/preprocess_text.py` **Execute at the project root directory.**

> Note that pre-processing may take time.
