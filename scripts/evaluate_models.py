#!/usr/bin/env python
"""
============================================================================
Model Evaluation Script
============================================================================
This script evaluates model responses using GPT or Claude as the judge.
It compares responses from two different models/checkpoints and computes
win rates.

Usage:
    python evaluate_models.py --model_a <exp_name_a> --model_b <exp_name_b>

Example:
    python evaluate_models.py --model_a qwen18_dpo_base_ep6 --model_b qwen18_dpo_extend_ep6
============================================================================
"""

import os
import json
import argparse
from typing import Dict, List
from tqdm import tqdm
import time

# Try to import API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not installed. Use 'pip install openai'")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic not installed. Use 'pip install anthropic'")


def gpt_evaluation_prompt_generator(prompt: str, response_A: str, response_B: str) -> str:
    """Generate the evaluation prompt for comparing two responses."""
    return f'''Given the history of multi-round chat, which response is more helpful?

History:
{prompt}

Response A: {response_A}

Response B: {response_B}

FIRST provide a one-sentence comparison of the two responses and explain which you feel is more helpful. \
SECOND, on a new line, state only "A" or "B" to indicate which response is more helpful. \
Your response should use the format:

Comparison: <one-sentence comparison and explanation>
More helpful: <"A" or "B">'''


def get_completion(msg: str, evaluator: str = 'claude', temp: float = 0.7) -> str:
    """Get completion from the specified evaluator (GPT or Claude)."""
    if evaluator == 'gpt':
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package not installed")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": 'system', "content": 'You are a helpful assistant.'},
                {'role': 'user', 'content': msg}
            ],
            temperature=temp,
            top_p=1,
        )
        return response.choices[0].message.content

    elif evaluator == 'claude':
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError("Anthropic package not installed")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model='claude-3-haiku-20240307',
            system='You are a helpful assistant.',
            messages=[{'role': 'user', 'content': msg}],
            temperature=temp,
            max_tokens=2048,
            top_p=1,
        )
        return response.content[0].text

    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")


def extract_responses(exp_name: str, base_path: str = "exp_results") -> Dict[str, List[str]]:
    """Extract responses from a generated response file."""
    response_dict = {"prompt": [], "response": []}

    # Try different possible file names
    possible_files = [
        os.path.join(base_path, exp_name, 'prob_test_gen_response.jsonl'),
        os.path.join(base_path, f'gen_{exp_name}', 'prob_test_gen_response.jsonl'),
    ]

    response_file = None
    for f in possible_files:
        if os.path.exists(f):
            response_file = f
            break

    if response_file is None:
        raise FileNotFoundError(
            f"Response file not found. Tried: {possible_files}"
        )

    with open(response_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = data['prompt'].strip('\n')
            tmp_response = data['response'].strip('\n')

            # Extract just the response part
            if prompt in tmp_response:
                response = tmp_response.split(prompt)[1].strip(' ')
            elif 'Assistant: ' in tmp_response:
                response = tmp_response.split('Assistant: ')[-1]
            else:
                response = tmp_response

            response_dict["prompt"].append(prompt)
            response_dict["response"].append(response)

    return response_dict


def evaluate_models(
    model_a: str,
    model_b: str,
    evaluator: str = 'claude',
    base_path: str = "exp_results",
    output_dir: str = "eval_results",
    max_response_len: int = 2000
) -> Dict[str, float]:
    """
    Evaluate two models against each other using an LLM judge.

    Args:
        model_a: Experiment name for model A
        model_b: Experiment name for model B
        evaluator: Which LLM to use as judge ('gpt' or 'claude')
        base_path: Base path for experiment results
        output_dir: Directory to save evaluation logs
        max_response_len: Maximum response length to evaluate

    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading responses from {model_a}...")
    res_a = extract_responses(model_a, base_path)

    print(f"Loading responses from {model_b}...")
    res_b = extract_responses(model_b, base_path)

    assert len(res_a['prompt']) == len(res_b['prompt']), \
        "Models have different number of responses"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Log file
    log_file = os.path.join(output_dir, f"{model_a}_vs_{model_b}_{evaluator}.txt")

    a_wins = 0
    b_wins = 0
    total = 0
    skipped = 0

    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Evaluation: [{model_a}] vs [{model_b}]\n")
        f.write(f"Evaluator: {evaluator}\n")
        f.write(f"{'='*60}\n\n")

        for i in tqdm(range(len(res_a['prompt'])), desc="Evaluating"):
            assert res_a['prompt'][i] == res_b['prompt'][i], \
                f"Prompt mismatch at index {i}"

            prompt = res_a['prompt'][i]
            resp_a = res_a['response'][i]
            resp_b = res_b['response'][i]

            # Skip very long responses
            if len(resp_a) > max_response_len or len(resp_b) > max_response_len:
                skipped += 1
                continue

            # Generate evaluation prompt and get response
            eval_prompt = gpt_evaluation_prompt_generator(prompt, resp_a, resp_b)

            try:
                judge_response = get_completion(eval_prompt, evaluator)

                # Record result
                f.write(f"#======== Question {i}: {prompt[:40]}...\n")
                f.write(f"Response A: {resp_a[:100]}...\n")
                f.write(f"Response B: {resp_b[:100]}...\n")
                f.write(f"Judge: {judge_response}\n\n")

                # Extract winner
                total += 1
                answer = judge_response.split(': ')[-1].strip()
                if answer == 'A':
                    a_wins += 1
                elif answer == 'B':
                    b_wins += 1

                # Small delay to avoid rate limiting
                time.sleep(0.5)

            except Exception as e:
                f.write(f"#======== Error at Question {i}: {str(e)}\n\n")
                continue

        # Summary
        if total > 0:
            win_rate_a = a_wins / (a_wins + b_wins) if (a_wins + b_wins) > 0 else 0
            win_rate_b = b_wins / (a_wins + b_wins) if (a_wins + b_wins) > 0 else 0

            summary = f"""
{'='*60}
EVALUATION SUMMARY
{'='*60}
Model A ({model_a}): {a_wins} wins ({win_rate_a:.1%})
Model B ({model_b}): {b_wins} wins ({win_rate_b:.1%})
Total evaluated: {total}
Skipped (too long): {skipped}
{'='*60}
"""
            f.write(summary)
            print(summary)

            return {
                'model_a': model_a,
                'model_b': model_b,
                'a_wins': a_wins,
                'b_wins': b_wins,
                'total': total,
                'skipped': skipped,
                'win_rate_a': win_rate_a,
                'win_rate_b': win_rate_b
            }

    return {}


def main():
    parser = argparse.ArgumentParser(description='Evaluate model responses using LLM judge')
    parser.add_argument('--model_a', type=str, required=True,
                        help='Experiment name for model A')
    parser.add_argument('--model_b', type=str, required=True,
                        help='Experiment name for model B')
    parser.add_argument('--evaluator', type=str, default='claude',
                        choices=['gpt', 'claude'],
                        help='Which LLM to use as judge (default: claude)')
    parser.add_argument('--base_path', type=str, default='exp_results',
                        help='Base path for experiment results')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory to save evaluation logs')

    args = parser.parse_args()

    # Change to src directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..', 'src'))

    evaluate_models(
        model_a=args.model_a,
        model_b=args.model_b,
        evaluator=args.evaluator,
        base_path=args.base_path,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
