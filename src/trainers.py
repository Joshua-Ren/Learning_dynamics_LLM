import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib
import pdb
from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards

def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size,
                sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens
                with a value of -100 are ignored.
                Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per 
                          (non-masked) token. Otherwise, return the sum of the 
                          log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log
        probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    logprob_logits = logits.log_softmax(-1)
    V = logprob_logits.shape[-1]
    per_token_logps = torch.gather(logprob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    # --------- Observe the argmax for each token
    labels_argmax = torch.argmax(logits, dim=-1)  # [B, M], argmax p(y*|chi_u^+/-)
    per_token_logps_argmax = torch.gather(logprob_logits, dim=2, index=labels_argmax.unsqueeze(2)).squeeze(2)
    #breakpoint()
    # ------ 2024-11-15 get all other metrics, e.g., expect_argmax, |A_o|_F, |p-e|_2
    prob_logits = logits.softmax(-1) # prob version of logits, [B, M, V], easy to get underflow, take care!!!
        # --------- expect_argmax, should be [B, M]
    per_token_prob_argmax = torch.exp(per_token_logps_argmax) #torch.gather(prob_logits, dim=2, index=labels_argmax.unsqueeze(2)).squeeze(2) #[B, M]
    per_token_prob_exceptargmax =  torch.ones_like(per_token_prob_argmax)* loss_mask - per_token_prob_argmax* loss_mask #[B, M]
    per_token_logp_exceptargmax = torch.log(per_token_prob_exceptargmax + 1e-100)
        # --------- |A_o|_F, should be [B, 1]
    #prob_norm = torch.norm(prob_logits, dim=-1) # [B, M, V] -> [B, M]
    prob_norm = torch.linalg.vector_norm(prob_logits, ord=2, dim=-1) # [B, M, V] -> [B, M], doing the same thing with previous line
    prob_norm = prob_norm * loss_mask # [B, M], all other dims are zeros
    prob_norm2_mean = torch.square(prob_norm.sum(-1) / loss_mask.sum(-1)) # [B, M] -> [B, 1]
    A_norm = torch.sqrt(V*prob_norm2_mean + (V-2)*torch.ones_like(prob_norm2_mean))  #[B, 1], align with the shape of all other metrics
        # ---------- |pi-e|_2, or 
    #breakpoint()
    e_oht = torch.nn.functional.one_hot(labels, num_classes=V) # [B, M, V]
    prob_gap_norm = torch.linalg.vector_norm(prob_logits - e_oht, ord=2, dim=-1) # [B, M, V] -> [B, M]
    prob_gap_norm = prob_gap_norm * loss_mask
    prob_gap2_mean = prob_gap_norm.sum(-1) / loss_mask.sum(-1)
        # --------- (p_label - 1), only the pull-up energy
    prob_label = torch.gather(prob_logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    prob_label_gap = torch.ones_like(prob_label) - prob_label # [B,M]
    prob_energy = (prob_label_gap*loss_mask).sum(-1) / loss_mask.sum(-1)
    #breakpoint()

    out_token  = (per_token_logps * loss_mask).sum(-1)  #[B, 1]
    out_argmax = (per_token_logps_argmax * loss_mask).sum(-1)
    out_except_argmax = (per_token_logp_exceptargmax * loss_mask).sum(-1)
    
    if average_log_prob:
        return out_token / loss_mask.sum(-1), (out_argmax / loss_mask.sum(-1), out_except_argmax / loss_mask.sum(-1), A_norm, prob_gap2_mean, prob_energy, labels_argmax)
    else:
        return out_token, (out_argmax, out_except_argmax, A_norm, prob_gap2_mean, prob_energy, labels_argmax)

def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch

class BasicTrainer(object):
    def __init__(
            self, policy: nn.Module, config: DictConfig, seed: int,
            run_dir: str, reference_model: Optional[nn.Module] = None,
            rank: int = 0, world_size: int = 1
    ) -> None:
        """A trainer for a language model, supporting either SFT training.

        If multiple GPUs are present, naively splits the model across them, 
        effectively offering N times available memory, but without any parallel 
        computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.prob_dicts = ['chosen', 'chosen_initial', 'chosen_gptsemantic', 'chosen_gptformat', # 'chosen_selfr'
                 'rejected', 'reject_gptsemantic', 'reject_gptformat',
                 'irr_train', 'irr_test', 'irr_hum',
                 'random_permute', 'random_nonhum']

        tokenizer_name_or_path = \
            config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs)
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
        )

        self.policy = policy
        self.reference_model = reference_model

        self.probtrain_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split='formal_prob_train',
            n_examples=500,
            shuffle=False,
            batch_size=config.eval_batch_size,
            silent=rank != 0,
        )
        self.probtrain_batches = list(self.probtrain_iterator)
        rank0_print(f'===========Loaded {len(self.probtrain_batches)} prob_train batches ')
        
        self.probtest_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split='formal_prob_test',
            n_examples=500,
            shuffle=False,
            batch_size=config.eval_batch_size,
            silent=rank != 0,
        )
        self.probtest_batches = list(self.probtest_iterator)
        rank0_print(f'========Loaded {len(self.probtest_batches)} prob_test batches ')

        if config.train_using_prob:
            self.train_iterator = get_batch_iterator(
                **data_iterator_kwargs,
                split="formal_prob_train",
                shuffle=True,
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
            )
        else:
            self.train_iterator = get_batch_iterator(
                **data_iterator_kwargs,
                split=config.train_split, #"train_dpo",
                shuffle=True,
                n_epochs=config.n_epochs,
                n_examples=config.n_examples,
                batch_size=config.batch_size,
                silent=rank != 0,
            )
        # self.train_batches = list(self.train_iterator)
        # rank0_print(f'===========Loaded {len(self.train_batches)} train batches ')


    def get_batch_samples(
            self, batch: Dict[str, torch.LongTensor],
            sample_flag = True
    ) -> Tuple[str, str]:
        """Generate samples from the policy for the given batch of inputs."""

        # FSDP generation according to
        # https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (
            FSDP.summon_full_params(
                self.policy, writeback=False, recurse=False
            ) if 'FSDP' in self.config.trainer else contextlib.nullcontext()
        )
        with ctx():
            policy_output = self.policy.generate(
                batch['prompt_input_ids'],
                attention_mask=batch['prompt_attention_mask'],
                max_length=self.config.max_length,
                do_sample=sample_flag,
                pad_token_id=self.tokenizer.pad_token_id
            )

        if self.config.loss.name in {'dpo', 'ipo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        
        policy_output = pad_to_length(
            policy_output, self.config.max_length, self.tokenizer.pad_token_id
        )
        policy_output = all_gather_if_needed(
            policy_output, self.rank, self.world_size
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        if self.config.loss.name in {'dpo', 'ipo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps, _ = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        loss_config: DictConfig,
        train=True,
        prob_set=None,
        force_sft=False
    ) -> Tuple[torch.FloatTensor, Dict[str, List]]:
        """Compute the SFT loss and other metrics for the given batch of inputs.
        """

        metrics = {}
        train_test = 'train' if train else 'eval'
        
        # if self.config.train_supervise=='rejected':
        #     chosen = 'rejected'
        #     print('@@@@@@@@@@@ Here we will use rejected sample as y+ @@@@@@@@@@@@@@@@@@@@@')
        # else:
        #     chosen = 'chosen'  

        
        if self.config.train_supervise is None:
            chosen = 'chosen'
        else:
            chosen = self.config.train_supervise

        if train:
            if loss_config.name in {'dpo', 'ipo'} and not force_sft:
                policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

                if loss_config.name == 'dpo':
                    loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
                elif loss_config.name == 'ipo':
                    loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
                else:
                    raise ValueError(f'unknown loss {loss_config.name}')

                losses, chosen_rewards, rejected_rewards = preference_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
                rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
                reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

                metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
                metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
                metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
                metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

                policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
                metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
                argmax_token=np.array([-1])

            else: #if loss_config.name == 'sft':
                policy_chosen_logits = self.policy(batch[f'{chosen}_input_ids'], attention_mask=batch[f'{chosen}_attention_mask']).logits.to(torch.float32)
                policy_chosen_logps, _ = _get_batch_logps(policy_chosen_logits, batch[f'{chosen}_labels'], average_log_prob=False)

                losses = -policy_chosen_logps

            policy_chosen_logps = all_gather_if_needed(
                policy_chosen_logps.detach(), self.rank, self.world_size
            )
            
            metrics[f'logps_{train_test}/chosen'] = \
                policy_chosen_logps.cpu().numpy().tolist()

            all_devices_losses = all_gather_if_needed(
                losses.detach(), self.rank, self.world_size
            )
            metrics[f'loss/{train_test}'] = \
                all_devices_losses.cpu().numpy().tolist() 
            loss_mean = losses.mean()
        else:
            if prob_set is not None:
                argmax_token=np.array([0])
                with torch.no_grad():
                    for k in self.prob_dicts:
                        policy_predict_logtis = self.policy(
                            input_ids=batch[f'{k}_input_ids'],
                            attention_mask=batch[f'{k}_attention_mask']
                        ).logits.detach().to(torch.float32)
                        policy_predict_logps, policy_argmax_logps = _get_batch_logps(policy_predict_logtis, batch[f'{k}_labels'],average_log_prob=False)
                        del policy_predict_logtis
                        metrics[f'logps_{train_test}_{prob_set}/{k}'] = \
                            policy_predict_logps.cpu().numpy().tolist()
                        metrics[f'argmax_prob_logits'] = policy_argmax_logps[0].cpu().numpy().tolist()
                        metrics[f'except_argmax_prob_logits'] = policy_argmax_logps[1].cpu().numpy().tolist()
                        metrics[f'{k}_A_o'] = policy_argmax_logps[2].cpu().numpy().tolist()
                        metrics[f'p_e'] = policy_argmax_logps[3].cpu().numpy().tolist()
                        metrics[f'energy'] = policy_argmax_logps[4].cpu().numpy().tolist()
                        argmax_token = policy_argmax_logps[5].squeeze().cpu().numpy()  # A [B, M] array that track the argmax of each token in response
                                                                                       # Annoying, remember to use validation_bath = 1 to get correct thing
            loss_mean = 0
        return loss_mean, metrics, argmax_token

    def evaluation(self, prob_set='prob_train'):
        if prob_set.lower()=='prob_train':
            data_batches = self.probtrain_batches
        elif prob_set.lower()=='prob_test':
            data_batches = self.probtest_batches
        else:
            raise ('only have prob_set naming prob_train or prob_test')

        self.policy.eval()
        all_eval_metrics = defaultdict(list)
        all_argmax_token = []
        for eval_batch in (tqdm.tqdm(data_batches, desc='Computing eval metrics') if self.rank == 0 else data_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)     
            with torch.no_grad():
                # ----- detail_eval_matrics is token-wise, for k, v... then each v[i] contains logp of M tokens
                _, eval_metrics, argmax_token = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False, prob_set=prob_set)                          
            all_argmax_token.append(argmax_token)  # argmax_token is [B, M], all_argmax_token is a list storing those argtokens with different length
            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)
                 
        # -------- Save the corresponding results
        logp_npy = np.zeros((1,1))
        mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)
        if self.rank==0:
            output_dir = os.path.join(self.config.save_path, f'{prob_set}_metrics.json')
            self.save_metrics(output_dir, mean_eval_metrics)
            print(mean_eval_metrics)
            if self.config.fine_evaluation:
                # --------- Save logp; first N are for different y, last four are argmax, expect_argmax, |A_o|, |p-e| for each example
                tmp = []       
                for k, v in all_eval_metrics.items():
                    tmp.append(v)
                logp_npy = np.array(tmp)
                #breakpoint()
        return logp_npy, all_argmax_token
                
    def evaluation_get_response(self, prob_set='prob_train_gen'):
        data_iterator_kwargs = dict(
                    names=self.config.datasets,
                    tokenizer=self.tokenizer,
                    max_length=self.config.max_length,
                    max_prompt_length=self.config.max_prompt_length,
                )
        probtest_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split=prob_set,
            n_examples=500,
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            silent=True,
        )
        data_batches = list(probtest_iterator)
        rank0_print(f'========Loaded {len(data_batches)} prob_test batches ')

        self.policy.eval()
        all_policy_samples = []
        for eval_batch in (tqdm.tqdm(data_batches, desc='Computing eval metrics') if self.rank == 0 else data_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
            with torch.no_grad():
                policy_samples, _ = self.get_batch_samples(local_eval_batch, sample_flag=True)  # For greedy decoding, convert this to False
                for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                    all_policy_samples.append({'prompt':prompt, 'response':sample})
        output_dir = os.path.join(self.config.save_path, f'{prob_set}_response.jsonl')
        with open(output_dir, 'a',newline='\n') as f:
            for i in range(len(all_policy_samples)):
                f.write(json.dumps(all_policy_samples[i]))
                f.write('\n')

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            self.policy.parameters(), lr=self.config.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(
                1.0, (step + 1) / (self.config.warmup_steps + 1)
            )
        )

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None
        logp_npy_all = []
        argmax_npy_all = []
        # reload_ref_required = True
        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} ' + 'train examples')
                logp_npy, argmax_npy = self.evaluation(prob_set='prob_train')  # [B,1] and [B, M]
                #breakpoint()
                if self.rank==0:
                    logp_npy_all.append(logp_npy)
                    argmax_npy_all.append(argmax_npy)
                #self.evaluation(prob_set='prob_test')
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()
            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(
                self.config.gradient_accumulation_steps
            ):
                global_microbatch = slice_and_move_batch_for_device(
                    batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank
                )
                local_microbatch = slice_and_move_batch_for_device(
                    global_microbatch, self.rank, self.world_size, self.rank
                )

                # if self.batch_counter < self.config.pre_sft_steps:
                #     loss, metrics, _ = self.get_batch_metrics(local_microbatch, self.config.loss, train=True, force_sft=True)
                # else:
                #     if reload_ref_required:
                #         self.reference_model = self.policy
                #         self.reference_model.eval()
                #         reload_ref_required = False
                loss, metrics, _ = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > \
                self.config.minimum_log_interval_secs:
                mean_train_metrics = {
                    k: sum(v) / len(v) for k, v in batch_metrics.items()
                }
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(
                    f'train stats after {self.example_counter} examples: ' + \
                    f'{formatted_dict(mean_train_metrics)}'
                )

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            # else:
            #     rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####
        # --- Train numpy results
        if self.config.fine_evaluation:
            output_dir = os.path.join(self.config.save_path, f'logp_npy_all_{self.config.train_supervise}.npy')
            output_dir_argmax = os.path.join(self.config.save_path, f'argmax_token_all_{self.config.train_supervise}.npy')
            #breakpoint()
            np.save(output_dir, np.array(logp_npy_all))
            np.save(output_dir_argmax, np.array(argmax_npy_all, dtype=object), allow_pickle=True)
            
        if self.config.save_ckp:
            output_dir = os.path.join(self.config.save_path)
            self.save(output_dir)


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.config.max_grad_norm
        ).item()

    def write_state_dict(
        self, step: int, state: Dict[str, torch.Tensor],
        metrics: Dict, filename: str, dir_name: Optional[str] = None
    ) -> None:
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def save_metrics(self, output_name=None, metrics=None):
        with open(output_name, 'a',newline='\n') as f:
            json.dump(metrics, f)
            f.write('\n')

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(
            self.example_counter,
            policy_state_dict,
            metrics,
            'policy.pt',
            output_dir
        )
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(
            self.example_counter,
            optimizer_state_dict,
            metrics,
            'optimizer.pt',
            output_dir
        )
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(
            self.example_counter,
            scheduler_state_dict,
            metrics,
            'scheduler.pt',
            output_dir
        )

class FSDPTrainer(BasicTrainer):
    def __init__(
        self,
        policy: nn.Module,
        config: DictConfig,
        seed: int,
        run_dir: str,
        reference_model: Optional[nn.Module] = None,
        rank: int = 0,
        world_size: int = 1
    ) -> None:
        """A trainer subclass that uses PyTorch FSDP to shard the model across
        multiple GPUs.
        
        This trainer will shard both the policy and reference model across all 
        available GPUs. Models are sharded at the block level, where the block 
        class name is provided in the config.
        """

        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )
        assert config.model.block_name is not None, \
            'must specify model.block_name ' + \
            '(e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={wrap_class},
        )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(
            torch, config.model.fsdp_policy_mp
        ) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(
            param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype
        )
        self.policy = FSDP(
            policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy
        )

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/
                #scaling-multimodal-foundation-models-in-torchmultimodal/
                #-with-pytorch-distributed/
                # first, verify we have FSDP activation support ready by 
                # importing:
                from \
                torch.distributed.algorithms._checkpoint.checkpoint_wrapper \
                import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print(
                    'Applying activation checkpointing wrapper to policy...'
                )
                apply_activation_checkpointing(
                    self.policy,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                    check_fn=check_fn
                )
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy,
           gathering the gradients across all GPUs.
        """
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            self.policy, StateDictType.FULL_STATE_DICT,
            state_dict_config=save_policy
        ):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                policy_state_dict,
                metrics,
                'policy.pt',
                output_dir
            )
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            self.policy,
            StateDictType.FULL_STATE_DICT,
            optim_state_dict_config=save_policy
        ):
            optimizer_state_dict = FSDP.optim_state_dict(
                self.policy, self.optimizer
            )

        if self.rank == 0:
            self.write_state_dict(
                self.example_counter,
                optimizer_state_dict,
                metrics,
                'optimizer.pt',
                output_dir
            )
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(
                self.example_counter,
                scheduler_state_dict,
                metrics,
                'scheduler.pt',
                output_dir
            )
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model 
        across multiple GPUs.

        Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling 
        is extremely slow, see 
        https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(
            policy, config, seed, run_dir, reference_model, rank, world_size
        )

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(
                reference_model, sharded=False
            )

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        self.write_state_dict(
            self.example_counter,
            policy_state_dict,
            metrics,
            'policy.pt',
            output_dir
        )
        del policy_state_dict
