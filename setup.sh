#!/bin/bash
conda env create -f environment.yml
source activate neural_embeddings_env
pip install -r ./WhisperSeg/requirements.txt