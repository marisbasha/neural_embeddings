import os
import json
import torch
import numpy as np
from tqdm import tqdm
from itertools import product
from torchmetrics.functional.pairwise import pairwise_cosine_similarity, pairwise_euclidean_distance
import gc
import yaml

def load_tensor(file_path):
    """
    Load a tensor from a file.

    Args:
        file_path (str): Path to the tensor file.

    Returns:
        numpy.ndarray: The loaded tensor as a NumPy array.
    """
    matrix = torch.load(file_path)
    return matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix

def split_in_history(dataset, batch_size, feature_type, distance_type):
    """
    Split the dataset into batches and prepare features for distance computation.

    Args:
        dataset (list): List of dataset items.
        batch_size (int): Size of each batch.
        feature_type (str): Type of feature to extract ('mels', 'embeddings', 'embeddings_first', or 'codecs').
        distance_type (str): Type of distance to compute.

    Returns:
        dict: A dictionary containing batched features.
    """
    history = {}
    size_array = set()
    
    # Process dataset in batches
    for p in tqdm(range(0, len(dataset), batch_size), desc=f'Load history {feature_type}'):
        mini = min(len(dataset), p + batch_size)
        batch = dataset[p:mini]
        features = []
        
        # Extract features based on feature_type
        for item in batch:
            if feature_type == 'mels':
                v = load_tensor(item['mel']).flatten()
            elif feature_type in ['embeddings', 'embeddings_first']:
                v = load_tensor(item['embedding'])
                v = v.flatten() if feature_type == 'embeddings' else v.squeeze()[:, 0]
            else:  # codecs
                v = load_tensor(item['codecs']).flatten()
            
            size_array.add(len(v))
            v = torch.tensor(v, dtype=torch.float16)
            if distance_type == 'spearman':
                v = v.argsort(dim=0).to(torch.int32)
            features.append(v)
        
        history[p] = features
    
    # Align feature sizes if necessary
    max_size = max(size_array)
    for p in tqdm(history.keys(), desc='Aligning and Converting'):
        dtype = torch.int32 if distance_type == 'spearman' else torch.float16
        aligned = torch.zeros((len(history[p]), max_size), dtype=dtype)
        for i, feat in enumerate(history[p]):
            aligned[i, :len(feat)] = feat
        history[p] = aligned
    
    return history

def compute_distances(dataset, batch_size, distance_type, feature_type, output_dir, subset):
    """
    Compute pairwise distances between dataset items.

    Args:
        dataset (list): List of dataset items.
        batch_size (int): Size of each batch for processing.
        distance_type (str): Type of distance to compute ('euclidean', 'cosine', or 'spearman').
        feature_type (str): Type of feature to use for distance computation.
        output_dir (str): Directory to save the output distance matrix.
        subset (str): Name of the dataset subset being processed.
    """
    history = split_in_history(dataset, batch_size, feature_type, distance_type)
    length = len(dataset)
    mat = np.zeros((length, length), dtype=np.float16)
    
    # Compute distances in batches
    for i in tqdm(range(0, length, batch_size), desc=f'Compute {distance_type} {feature_type}'):
        mini = min(length, i + batch_size)
        feature1 = history[i].cuda() if distance_type != 'spearman' else history[i]
        
        for j in range(0, length, batch_size):
            minj = min(length, j + batch_size)
            if i > j:
                # Use symmetry of distance matrix to avoid redundant computations
                mat[i:mini, j:minj] = mat[j:minj, i:mini].T
                continue
            
            feature2 = history[j].cuda() if distance_type != 'spearman' else history[j]
            
            # Compute distances based on distance_type
            if distance_type == 'euclidean':
                output = pairwise_euclidean_distance(feature1.double(), feature2.double(), zero_diagonal=True)
                output = output.cpu().half().numpy()
            elif distance_type == 'cosine':
                output = pairwise_cosine_similarity(feature1.double(), feature2.double())
                output = torch.divide(torch.ones_like(output) - output, 2)
                output = output.cpu().half().numpy()
            else:  # spearman
                output = np.corrcoef(feature1, feature2, rowvar=True)
                inn = feature1.shape[0]
                outt = inn + feature2.shape[0]
                output = output[:inn, inn:outt] # Because np.corrcoef concats both arrays.
                output = (1 - output) / 2
            
            mat[i:mini, j:minj] = output
            del output
            del feature2
        del feature1
    
    del history
    gc.collect()
    
    # Save the computed distance matrix
    output_file = os.path.join(output_dir, f'{subset}--{distance_type}++{feature_type}.pt')
    torch.save(mat, output_file, pickle_protocol=4)
    del mat
    gc.collect()

def init_results(ds, distances, features):
    permutations = list(product(distances, features))
    metadata = {
        'speakers': set(),
        'labels': set(),
    }
    indicies = []
    wav_loc = []
    for i in ds:
        indicies.append(str(i['speaker'])+'////'+str(i['label']))
        wav_loc.append(i['waveform'])
        metadata['speakers'].add(i['speaker'])
        metadata['labels'].add(i['label'])
    metadata['speakers'] = list(metadata['speakers'])
    metadata['labels'] = list(metadata['labels'])
    metadata['permutations'] = permutations
    return metadata, indicies, distances, features, wav_loc

def main():
    """
    Main function to orchestrate the distance computation process.
    """
    # Load configuration
    with open('reproducibiliy_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    data_dir = config['data_dir']
    subsets = config['subsets']
    batch_size = config['batch_size']
    distances = config['distances']
    features = config['features']

    # Process each subset
    for subset in subsets:
        dataset_path = os.path.join(data_dir, subset, 'features', 'dataset.json')
        output_dir = os.path.join(data_dir, subset, 'results')
        save_path = os.path.join(output_dir, f"results.json")
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.isfile(dataset_path):
            print(f'Run extract features for {subset} before computing distances')
            continue
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        metadata, indicies, distances, features, wav_loc = init_results(dataset, distances=distances, features=features)
        with open(save_path, 'w') as f:
            f.write(json.dumps({
            'metadata': metadata,
            'indicies': indicies,
            'wav_loc': wav_loc
        }))
        del metadata
        del indicies
        for distance_type in distances:
            for feature_type in features:
                try:
                    compute_distances(dataset, batch_size, distance_type, feature_type, output_dir, subset)
                except Exception as e:
                    print(f"Error processing {subset} with {distance_type} distance and {feature_type} feature: {str(e)}")
        
        print(f"Finished processing subset {subset}")

if __name__ == "__main__":
    main()