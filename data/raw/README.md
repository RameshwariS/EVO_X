Place real CICIDS2017 CSV files here (download from
https://www.unb.ca/cic/datasets/ids-2017.html).

`main.py` will automatically detect and use any `*.csv` file placed in this
folder. If this folder is empty, the pipeline falls back to a synthetic,
schema-compatible dataset (see ../../src/synthetic_data.py) so the code can
still be run and tested.
