import os
import json
import torch
import numpy as np
from tqdm import tqdm
import yaml
from collections import Counter
import gc
from itertools import product
from typing import Dict, List, Any

def load_tensor(file_path):
    matrix = torch.load(file_path)
    return matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix

def get_class(label, speakers):
    if speakers:
        return '////'.join(label.split('////')[1:])
    return label.split('////')[-1]

def get_splits(indicies, labels, speakers, info):
    loops = {}
    loops_label = {}
    split_indicies = {}
    split_labels = {}
    percentage_labels = {}
    percentage_used = {}

    compute_speakers = bool(speakers)
    counts = Counter(get_class(label, compute_speakers) for label in indicies)
    counts_filtered = [(label, count) for label, count in counts.items() if count > 5]
    labels = [label for label, _ in counts_filtered]
    
    if 'remove_first' in info:
        counts_filtered = counts_filtered[info['remove_first']:]
        labels_filtered = labels[info['remove_first']:]
    else:
        labels_filtered = labels

    usable_indexes = [(index, value) for index, value in enumerate(indicies) if get_class(value, compute_speakers) in labels_filtered]
    index_to_label = {index: value for index, value in usable_indexes}
    label_indexes = {label: [index for index, value in usable_indexes if get_class(value, compute_speakers) == label] for label in labels_filtered}

    if info.get('top_percent'):
        for percent in info['top_percent']:
            min_count = int(len(labels_filtered) * percent[0] / 100)
            max_count = int(len(labels_filtered) * percent[1] / 100)
            trimmed_labels = labels_filtered[min_count:max_count]
            percentage_labels[max_count] = [label for label, _ in counts_filtered if label in trimmed_labels]
            percentage_used[max_count] = percent
        
        for max_count, values in percentage_labels.items():
            split_indicies[f'percent-{percentage_used[max_count]}'] = [index for index, value in usable_indexes if get_class(value, compute_speakers) in values]
            split_labels[f'percent-{percentage_used[max_count]}'] = list(set(values))

    split_indicies['all'] = [index for index, _ in usable_indexes]
    split_labels['all'] = list(set([get_class(value, compute_speakers) for _, value in usable_indexes]))
            

    loops['splits'] = split_indicies
    loops_label['splits'] = split_labels

    return loops, loops_label, label_indexes, usable_indexes, index_to_label, counts_filtered

def compute_fvalue(matrix, indicies_considered, wrds, label_indexes, desc):
    f_values = {}
    tens = {label: torch.tensor(list(set(indicies_considered) & set(label_indexes[label])), dtype=torch.int32) for label in wrds}

    for label1 in tqdm(wrds, desc=desc):
        indicies1_tensor = tens[label1]
        if indicies1_tensor.size(0) == 0:
            continue
        f_values[label1] = {}
        
        self_similarities = matrix[indicies1_tensor][:, indicies1_tensor]
        self_similarities[torch.eye(self_similarities.shape[0], dtype=bool)] = float('nan')
        mean_self = torch.nanmean(self_similarities).item()
        std = torch.std(self_similarities[torch.logical_not(torch.isnan(self_similarities))]).item()
        
        f_values[label1][label1] = {
            'self': True,
            'std': std,
            'mean': mean_self,
            'f_value': 1,
        }

        for label2 in wrds:
            if label2 == label1:
                continue
            indicies2_tensor = tens[label2]
            if indicies2_tensor.size(0) == 0:
                continue
            between_labels = matrix[indicies1_tensor][:, indicies2_tensor]
            
            worse_calc = between_labels.clone()
            worse_calc[torch.isnan(worse_calc)] = float('-inf')
            worse_values, worse = torch.topk(worse_calc.flatten(), k=4)
            unraveled = np.unravel_index(worse.cpu().numpy(), worse_calc.shape)
            most_dissimilar = [(indicies1_tensor[unraveled[0][i]].item(), indicies2_tensor[unraveled[1][i]].item(), worse_values[i].item()) for i in range(4)]
            
            best_calc = between_labels.clone()
            best_calc[torch.isnan(best_calc)] = float('inf')
            best_values, best = torch.topk(best_calc.flatten(), k=4, largest=False)
            unraveled = np.unravel_index(best.cpu().numpy(), best_calc.shape)
            most_similar = [(indicies1_tensor[unraveled[0][i]].item(), indicies2_tensor[unraveled[1][i]].item(), best_values[i].item()) for i in range(4)]

            mean_inter = torch.nanmean(between_labels).item()
            stdBet = torch.std(between_labels[torch.logical_not(torch.isnan(between_labels))]).item()

        
            f_values[label1][label2] = {
                'self': False,
                'std': stdBet,
                'mean': mean_inter,
                'f_value': mean_inter / (mean_self + 1e-9),
                'most_similar': most_similar,
                'most_dissimilar': most_dissimilar,
                'count1': len(indicies1_tensor),
                'count2': len(indicies2_tensor),        
            }

    return f_values

def compute_accuracy(matrix, indicies_considered, index_to_label, desc, kss, compute_speakers):
    label_accs = {}
    most_common = {}
    ic = torch.tensor(indicies_considered, dtype=torch.int32)
    matrix = matrix[ic][:, ic]
    sorted_matrix = torch.argsort(matrix, dim=1)
    label_values = [[index_to_label[indicies_considered[i]] for i in row] for row in sorted_matrix]

    for i in tqdm(range(len(indicies_considered)), desc=desc):
        label = get_class(index_to_label[indicies_considered[i]], compute_speakers)
        sorted_values = label_values[i]
        label_distance = sorted_values[1:]
        num_instances = sum(1 for w in label_distance if get_class(w, compute_speakers) == label)
        ks = [min(k, num_instances) for k in kss if k <= num_instances]
        
        if label not in label_accs:
            label_accs[label] = {}
            most_common[label] = {'best': {'acc': -1, 'list': [], 'element': 0}, 'worst': {'acc': 2, 'list': [], 'element': 0}}
        
        real_indicies = [indicies_considered[idx] for idx in sorted_matrix[i, 1:11]]
        tp = sum(1 for s in label_distance[:10] if get_class(s, compute_speakers) == label) / 10

        if tp > most_common[label]['best']['acc']:
            most_common[label]['best'] = {'acc': tp, 'list': real_indicies, 'element': indicies_considered[i]}
        if tp < most_common[label]['worst']['acc']:
            most_common[label]['worst'] = {'acc': tp, 'list': real_indicies, 'element': indicies_considered[i]}

        for k in ks:
            if k not in label_accs[label]:
                label_accs[label][k] = []
            tp = sum(1 for s in label_distance[:k] if get_class(s, compute_speakers) == label) / k
            label_accs[label][k].append(tp)

    for label, values in label_accs.items():
        for k, val in values.items():
            vl = torch.tensor(val)
            vlnotnan = vl[~torch.isnan(vl)]
            label_accs[label][k] = {
                'mean': vl.nanmean().item(),
                'std': torch.std(vlnotnan).item(),
                'acc': vlnotnan.sum().item() / len(vlnotnan),
            }

    return label_accs, most_common

def save_json(where, filename, value):
    os.makedirs(where, exist_ok=True)
    with open(os.path.join(where, filename), 'w') as f:
        json.dump(value, f)

def ensure_torch_tensor(matrix: Any) -> torch.Tensor:
    if not isinstance(matrix, torch.Tensor):
        if isinstance(matrix, np.ndarray):
            return torch.from_numpy(matrix)
        elif isinstance(matrix, list):
            return torch.tensor(matrix)
        else:
            raise TypeError(f"Unsupported type for matrix conversion: {type(matrix)}")
    return matrix
    
def process_permutation(distance, feature, loops, loops_label, store, typed, label_indexes, stats_store, index_to_label, info):
    nn = f'{distance}++{feature}'
    matrix = torch.load(f'{store}/{typed}--{nn}.pt')
    matrix = ensure_torch_tensor(matrix)
    if matrix is None:
        print(f'empty {nn}')
        return
    print(f"{distance} {feature}, shape: {matrix.shape}")
    print(f"Min value: {torch.min(matrix).item()}, Max value: {torch.max(matrix).item()}")

    for split_type, split in loops.items():
        for split_element, indicies_split in split.items():
            saved_speaker = f'{split_type}--{split_element}'

            f_values = compute_fvalue(
                matrix=matrix,
                indicies_considered=indicies_split,
                wrds=loops_label[split_type][split_element],
                label_indexes=label_indexes,
                desc=f'Fvalue for {split_type} {split_element} and {nn} for type {typed}'
            )
            save_json(
                where=f'{stats_store}/{split_type}/',
                filename=f'fvalue--{typed}--{saved_speaker}--{nn}.json',
                value=f_values
            )
            del f_values
            gc.collect()

            label_accs, most_common = compute_accuracy(
                matrix=matrix,
                indicies_considered=indicies_split,
                index_to_label=index_to_label,
                desc=f'Accuracy for {split_type} {split_element} and {nn} for type {typed}',
                kss=info['k_values'],
                compute_speakers=info['compute_speakers']
            )
            save_json(
                where=f'{stats_store}/{split_type}/',
                filename=f'accuracy--{typed}--{saved_speaker}--{nn}.json',
                value=label_accs
            )
            del label_accs

            save_json(
                where=f'{stats_store}/{split_type}/',
                filename=f'most_common--{typed}--{saved_speaker}--{nn}.json',
                value=most_common
            )
            del most_common
            gc.collect()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path) as f:
        return json.load(f)

def process_dataset(config: Dict[str, Any], dataset: Dict[str, Any], info: Dict[str, Any], 
                    store: str, stats_store: str, path: str) -> None:
    speakers = dataset['metadata']['speakers'] if info['compute_speakers'] else False
    labels = dataset['metadata']['labels']
    indicies = dataset['indicies']

    loops, loops_label, label_indexes, usable_indexes, index_to_label, counts = get_splits(
        indicies=indicies,
        labels=labels,
        speakers=speakers,
        info=info,
    )

    save_json(
        where=stats_store,
        filename='count.json',
        value=counts
    )

    permutations = list(product(config['distances'], config['features']))
    for distance, feature in permutations:
        process_permutation(distance, feature, loops, loops_label, store, path, label_indexes, 
                            stats_store, index_to_label, info)
        
def main(config_path: str):
    config = load_config(config_path)
    init_path = config['data_dir']
    
    for path in config['subsets']:
        info = config['statistics'][path]
        store = f'{init_path}/{path}/results/'
        stats_store = f'{init_path}/{path}/stats/'
        create_directory(stats_store)
        
        dataset_path = f'{store}/results.json'
        dataset = load_json(dataset_path)
        print(dataset.keys())
        process_dataset(config, dataset, info, store, stats_store, path)
        
        gc.collect()

if __name__ == "__main__":
    config_path = "./reproducibiliy_config.yaml"
    main(config_path)