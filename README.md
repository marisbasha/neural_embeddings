# Benchmarking embeddings for retrieval and separation of vocalizations in humans and songbirds


## Install Environment and dependencies
```bash 
conda env create -f environment.yml
```
### Install either requirements_windows (for Windows) or requirements (for UNIX) for WhisperSeg
```bash 
pip install -r ./WhisperSeg/requirements.txt
```
```bash 
pip install -r ./WhisperSeg/requirements_windows.txt
```
### Or simply run the bash script
```bash 
sh setup.sh
```
## Reproducibility
To reproduce the results obtained in the paper, first modify ```reproducibility_config.yaml``` with your specific directories. Then running:
```bash 
python reproducibility_extract_features.py
```
will extract the mel, Whisper embeddings, and Encodec codes for all subsets.

Then:
```bash 
python reproducibility_compute_distances.py
```
will compute the distance metrics for all the permutations.

Lastly:
```bash 
python reproducibility_statistics.py
```
will create json files for accuracy and f-value.
## Contact
Anonymus


