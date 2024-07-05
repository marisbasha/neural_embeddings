import os
import json
import yaml
import shutil
import secrets
import string
from tqdm import tqdm
import numpy as np
import torch
import torchaudio
import librosa
from datasets import load_dataset, Audio
import ctranslate2
from encodec import EncodecModel
from encodec.utils import convert_audio
import sys
from scipy.io import wavfile

# Add WhisperSeg to the Python path
sys.path.append('./WhisperSeg')
from WhisperSeg.model import WhisperSegmenterFast

# Load configuration from YAML file
with open('reproducibiliy_config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Extract configuration values
hf_dataset = config['hf_dataset']
data_dir = config['data_dir']
plot_dir = config['plot_dir']
whisperseg_config = config['whisperseg_config']
subsets = config['subsets']

# Load the dataset
dataset = load_dataset(hf_dataset, split='train')

# Initialize models
segmenter = WhisperSegmenterFast("Systran/faster-whisper-large-v2", device="cpu")
encodec_model = EncodecModel.encodec_model_24khz()

def generate_random_id(length):
    """
    Generate a random ID of specified length.
    
    Args:
        length (int): Length of the random ID.
    
    Returns:
        str: Random ID consisting of ASCII letters and digits.
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_features(audio, sr, min_freq, spec_time_step, num_trials):
    """
    Generate mel spectrogram, whisper embedding, and encodec codecs for given audio.
    
    Args:
        audio (np.array): Audio signal.
        sr (int): Sampling rate of the audio.
        min_freq (float): Minimum frequency for spectrogram.
        spec_time_step (float): Time step for spectrogram.
        num_trials (int): Number of trials for WhisperSeg.
    
    Returns:
        tuple: Mel spectrogram, whisper embedding, and encodec codecs.
    """
    # Generate features using WhisperSeg
    ftr = segmenter.get_sliced_audios_features(audio, sr, min_freq, spec_time_step, num_trials)
    features = ctranslate2.StorageView.from_array(np.asarray([ftr[0][2]]))
    mel = ftr[0][2]
    encoded = segmenter.model_list[0].encode(features)
    embedding = torch.tensor(np.array(encoded).tolist(), device="cpu")
    
    # Process audio for EnCodec
    audio = torch.tensor(audio, dtype=torch.float32)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=encodec_model.sample_rate)
    resampled_audio = resampler(audio)
    if len(resampled_audio.shape) == 1:
        audio_tensor = resampled_audio.unsqueeze(0)  # Add channel dimension for mono audio

    # Convert tensor for torchaudio processing
    wav = convert_audio(audio_tensor, sr, encodec_model.sample_rate, encodec_model.channels)
    wav = wav.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        encoded_frames = encodec_model.encode(wav)
    codecs = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    
    return mel, embedding, codecs

def process_subset(subset, config):
    """
    Process a subset of the dataset, generating and saving features.
    
    Args:
        subset (str): Name of the subset to process.
        config (dict): Configuration for processing the subset.
    """
    # Create directory structure for features
    features_dir = os.path.join(data_dir, subset, 'features')
    os.makedirs(features_dir, exist_ok=True)
    for subdir in ['spectrograms', 'whisper_embeddings', 'encodec_codecs', 'raw_data']:
        os.makedirs(os.path.join(features_dir, subdir), exist_ok=True)

    processed_data = []
    subset_data = dataset.filter(lambda s: s['subset'] == subset)

    for item in tqdm(subset_data, desc=f"Processing {subset}"):
        try:
            random_id = generate_random_id(32)
            mel, embedding, codecs = generate_features(
                item['audio']['array'],
                item['audio']['sampling_rate'],
                config['min_freq'], 
                config['spec_time_step'], 
                config['num_trials']
            )

            # Define file paths
            mel_file = os.path.join(features_dir, 'spectrograms', f'{random_id}.pt')
            embedding_file = os.path.join(features_dir, 'whisper_embeddings', f'{random_id}.pt')
            codecs_file = os.path.join(features_dir, 'encodec_codecs', f'{random_id}.pt')
            waveform_file = os.path.join(features_dir, 'raw_data', f'{random_id}.wav')
            
            # Save features
            torch.save(mel, mel_file)
            torch.save(embedding, embedding_file)
            torch.save(codecs, codecs_file)
            
            # Save raw audio
            audio_array = item['audio']['array'].astype(np.float32)
            audio_array /= np.max(np.abs(audio_array))
            wavfile.write(waveform_file, item['audio']['sampling_rate'], audio_array)

            # Add processed item to dataset
            processed_data.append({
                'speaker': item.get('speaker', False),
                'label': item['label'],
                'mel': mel_file,
                'embedding': embedding_file,
                'codecs': codecs_file,
                'waveform': waveform_file
            })

        except Exception as e:
            print(f'Error processing item in {subset}: {e}')

    # Save processed dataset
    with open(os.path.join(features_dir, 'dataset.json'), 'w') as f:
        json.dump(processed_data, f)

# Process specified subsets
for subset in subsets:
    if subset in whisperseg_config['songbirds']['datasets']:
        config = whisperseg_config['songbirds']
    elif subset in whisperseg_config['humans']['datasets']:
        config = whisperseg_config['humans']
    else:
        print(f"Skipping unknown subset: {subset}")
        continue
    
    process_subset(subset, config)

print("Processing complete!")