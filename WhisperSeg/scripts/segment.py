import sys
import os
# sys.path.insert(0, os.path.dirname(os.getcwd()))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
from model import WhisperSegmenter, WhisperSegmenterFast
import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--audio_path")
    parser.add_argument("--csv_save_path" )
    parser.add_argument("--device", help="cpu or cuda", default = "cuda")
    parser.add_argument("--device_ids", help="a list of GPU ids", type = int, nargs = "+", default = [0,])
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--min_frequency", default= None, type=int)
    parser.add_argument("--spec_time_step", default= None, type=float)
    parser.add_argument("--num_trials", default= 3, type=int)
    args = parser.parse_args()

    assert args.csv_save_path.endswith(".csv") or args.csv_save_path == "buffer", "csv_save_path must ends with .csv or be 'buffer'"
    
    try:
        segmenter = WhisperSegmenterFast( args.model_path, device = args.device, device_ids = args.device_ids )
    except:
        segmenter = WhisperSegmenter( args.model_path, device = args.device, device_ids = args.device_ids )

    if args.audio_path == '-':
        audio_buffer = io.BytesIO(sys.stdin.buffer.read())
        audio, sr = librosa.load(audio_buffer, sr=None)
    else:
        audio, sr = librosa.load(args.audio_path, sr=None)
    
    res = segmenter.segment( audio, sr, min_frequency = args.min_frequency, spec_time_step = args.spec_time_step, num_trials = args.num_trials, batch_size = args.batch_size )
    df = pd.DataFrame( res )

    if args.csv_save_path == "buffer":
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        print( csv_buffer.getvalue() )  ## print the csv content to Stdout IO buffer, so that it can be captured by other process
    else:
        df.to_csv(args.csv_save_path, index=False)

    