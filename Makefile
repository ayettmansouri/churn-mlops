PYTHON  ?= python
JUPYTER ?= jupyter
EXEC_DIR := reports/executed
NB_EXEC  := $(JUPYTER) nbconvert --to notebook --execute --output-dir $(EXEC_DIR) --ExecutePreprocessor.timeout=1800

.DEFAULT_GOAL := help
.PHONY: help install eda data train clean

help:
	@echo "make install  - install dependencies"
	@echo "make eda      - run the exploratory analysis notebook"
	@echo "make data     - load, validate, clean, split, preprocess"
	@echo "make train    - full training pipeline (currently: data + preprocessing)"
	@echo "make clean    - remove generated artifacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

eda:
	$(NB_EXEC) notebooks/01_eda.ipynb

data:
	$(NB_EXEC) notebooks/02_data_pipeline.ipynb

# Later days add notebooks here (03 training + MLflow, ...)
train: data

clean:
	rm -rf reports/executed reports/figures data/processed/*.csv models/*.joblib models/*.json
