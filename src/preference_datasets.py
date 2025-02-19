import datasets
import json
import tqdm
import random
import torch
from utils import TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union
import transformers


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def get_hh(
    split: str,
    silent: bool = False,
    data_dir: str = None,
) -> List[Dict[str, str]]:
    """Load the Anthropic Helpful-Harmless dataset from a local file and
       convert it to the necessary format.

       Note, this dataset is not the original version on Huggingface, but
       has been preprocessed to add a few more synthetic responses.
    
       The dataset is converted to a list of dictionaries with the following 
       structure:
       [
           {
                'prompt': str,
                'chosen': str,
                'rejected': str,
                'random': str,
                'paraphrase': str,
                'variant': str,
                'nonresponse': str,
           },
           ...
       ]

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    data = []
    with open(f'{data_dir}/{split}.jsonl', 'r') as f:
        for line in tqdm.tqdm(
            f, 
            desc=f'Loading HH dataset ({split} split) from {data_dir}...',
            disable=silent
        ):
            data.append(json.loads(line))
    print('done')
    return data


def get_dataset(
        name: str,
        split: str,
        silent: bool = False,
        data_dir: str = None,
):
    """Load the given dataset by name. Supported by default are 'ultrafb', 'hh'."""
    if name == 'hh':
        data = get_hh(split, silent=silent, data_dir='./data/helpful-base')
        print("==================Got the dataset%s"%split)
    elif name == 'ultrafb':
        data = get_hh(split, silent=silent, data_dir='./data/ultrafeedback')
        print("==================Got the dataset%s"%split)        
    else:
        raise ValueError(f"Unknown dataset '{name}'")
    # assert set(data[0].keys()) == \
    #     {'prompt', 'chosen', 'rejected', 'random', 'paraphrase', 'variant',
    #      'nonresponse'}, f"Unexpected keys in dataset: {list(data[0].keys())}"

    return data


def get_collate_fn(
    tokenizer
) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are 
       lists of ints [tokens] or strings [the original texts]) and returns a 
       batch of examples, PyTorch tensors padded to the maximum length. Strings 
       are passed through.
    """
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') \
                or k.endswith('_labels'):
                # adapted from https://stackoverflow.com/questions/73256206
                if 'prompt' in k:  
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                # for the prompt, flip back so padding is on left side
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def tokenize_element_in_batch(
    element: str,
    tokenizer: Callable[[str], Dict[str, List[int]]],
    add_eos: bool = True
) -> Dict[str, List[int]]:
    """Tokenize a single element in a batch.
    
    Args:
        element: the element to tokenize.
        truncation_mode: to truncate the start or end of the sequence.
        tokenizer: the tokenizer to use.
        max_length: the maximum length of the sequence, or None to not truncate.
    
    Returns:
        t
    """
    tokens = tokenizer(element, add_special_tokens=False)

    assert tokenizer.eos_token_id not in tokens['input_ids'], \
        f"Element contains EOS token: {element}"

    if add_eos:
        tokens['input_ids'].append(tokenizer.eos_token_id)
        tokens['attention_mask'].append(1)

    return tokens


def tokenize_batch_element(
    data: Dict[str, str],
    truncation_mode: str,
    tokenizer,
    max_length: int,
    max_prompt_length: int
) -> Dict[str, Union[str, List[int]]]:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle
       the truncation in case the prompt + chosen or prompt + rejected 
       responses is/are too long. First we truncate the prompt; if we're still 
       too long, we truncate the chosen/rejected/paraphrase/variant/random/
       nonresponse.

       We also create the labels for the all responses, which are of length 
       equal to the sum of the length of the prompt and the chosen/rejected
       response, with -100 for the prompt tokens.

    Args:
        data: the batch data to tokenize.
        truncation_mode: to truncate the start or end of the sequence.
        tokenizer: the tokenizer to use.
        max_length: the maximum length of the sequence (prompt + response).
        max_prompt_length: the maximum length of the prompt.

    Returns:
        batch: the tokenized batch.
    """
    tokenized_data = {}
    for k, v in data.items():
        tokenized_data[k] = tokenize_element_in_batch(
            v, tokenizer, add_eos=(k != 'prompt')
        )

    longest_response_length = max(
        [len(v['input_ids']) for v in tokenized_data.values()]
    )

    # if combined sequence is too long, truncate the prompt
    if len(tokenized_data['prompt']['input_ids']) + longest_response_length \
        > max_length:
        if truncation_mode == 'keep_start':
            tokenized_data['prompt'] = {
                k: v[:max_prompt_length] \
                    for k, v in tokenized_data['prompt'].items()
            }
        elif truncation_mode == 'keep_end':
            tokenized_data['prompt'] = {
                k: v[-max_prompt_length:] \
                    for k, v in tokenized_data['prompt'].items()
            }
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(tokenized_data['prompt']['input_ids']) + longest_response_length \
        > max_length:
        for k, v in tokenized_data.items():
            if k == 'prompt':
                continue
            tokenized_data[k] = {
                k: v[:max_length - len(tokenized_data['prompt']['input_ids'])] \
                    for k, v in tokenized_data[k].items()
            }

    # Create labels
    sequences = {}
    for k, v in tokenized_data.items():
        if k == 'prompt':
            continue
        sequences[k] = {}
        sequences[k]['token_ids'] = \
            tokenized_data['prompt']['input_ids'] + v['input_ids']
        sequences[k]['attention_mask'] = \
            tokenized_data['prompt']['attention_mask'] + v['attention_mask']
        sequences[k]['labels'] = sequences[k]['token_ids'][:]
        sequences[k]['labels'][:len(tokenized_data['prompt']['input_ids'])] = \
            [-100] * len(tokenized_data['prompt']['input_ids']) 
        # -100 is the ignore index for cross-entropy loss

    batch = {}

    for k, v in data.items():
        if k == 'prompt':
            continue
        batch[f'{k}_response_only'] = v
    batch['prompt'] = data['prompt']

    for k in data.keys():
        if k == 'prompt':
            batch[f'{k}_input_ids'] = tokenized_data[k]['input_ids']
            batch[f'{k}_attention_mask'] = tokenized_data[k]['attention_mask']
        else:
            batch[f'{k}_input_ids'] = sequences[k]['token_ids']
            batch[f'{k}_attention_mask'] = sequences[k]['attention_mask']
            batch[f'{k}_labels'] = sequences[k]['labels']

    return batch


def get_batch_iterator(
    names: List[str],
    tokenizer,
    split: str = 'train',
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 128,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed:int = 0,
    silent: bool = False,
    data_dir: Optional[str] = './data/helpful-base'
) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        n_epochs: Number of epochs to run for. This or n_examples must be 
            specified.
        n_examples: Number of examples to run for. This or n_epochs must be 
            specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, \
        "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for name in [names]:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            for data in get_dataset(name, split, silent, data_dir):
                flat_data.append((data, truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(int(next(permutation_seeds))):
                random.shuffle(flat_data)

        batch = []
        for data, truncation_mode in flat_data:
            if done:
                break
            batch_element = tokenize_batch_element(
                data, truncation_mode, tokenizer, max_length, max_prompt_length
            )
            batch.append(batch_element)
            example_idx += 1
            if len(batch) == batch_size:
                yield collate_fn(batch)
                if n_examples is not None and example_idx >= n_examples:
                    if not silent:
                        print(
                            f'Finished generating {n_examples} examples ' + \
                            f'on{split} split'
                        )
                    done = True
                batch = []
        if done:
            break

    epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True
