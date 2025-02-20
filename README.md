# Learning Dynamics of LLM Finetuning

This repository is a reference implementation of [Learning Dynamics of LLM Finetuning](https://openreview.net/forum?id=tPNHOoZFl9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)) (ICLR 2025 Oral).

## TL;DR: 
Focusing on the interaction between training samples, the paper studies different LLM finetuning mechanisms using a unified framework.

#### About Learning Dynamics

1. It focuses on how learning example $x_o$ influences the model's confidence on $x_u$. In the context of LLM, it describes how the model's confidence in different $[x_u, y']$ changes after learning $[x_u,y_u^+]$.
2. It is a qualitative and microcosmic observation,  offering complementary insights into how specific training instances shape the model’s behavior. This complements the common analysis focusing on the model's convergence.
3. Its decomposition has a shape like $A(x_o)K(x_o,x_u)G(x_u)$, where $K$ is an eNTK term measuring the similarity between $x_u$ and $x_o$. $G$ is determined by loss function and supervisory signal.
4. The negative gradient and squeezing effect is a general phenomenon. It is not necessarily something to be avoided; rather, we must be cautious when forcing the model to forget information it is already uncertain about, otherwise, we might have this :-)

<div align=center><img src="https://github.com/Joshua-Ren/Learning_dynamics_LLM/blob/main/something_interesting/squeezing_effect.png" width="260"/><img src="https://github.com/Joshua-Ren/Learning_dynamics_LLM/blob/main/something_interesting/doutub_gif.gif" width="360"/></div>

#### About LLM's Finetuning

1. The framework amortizes the auto-regressive nature by integrating it into the definition of eNTK, leading to more concise non-token-wise modeling. 
However, this approximation relies on several assumptions, such as a relatively stable eNTK and causal masking (as discussed in the paper).
Consequently, it effectively explains algorithms trained on off-policy samples using the 'teacher-forcing' method.

2. The general principles could also apply to on-policy methods—or even RL-based approaches—but token-wise modeling is more suitable for them. We are currently exploring this direction, so please stay tuned.
3. The experimental results align well with our analysis. However, *while the squeezing effect is a primary factor, it is not the sole reason why off-policy DPO reduces overall confidence.*
   The interaction between positive and negative gradients is more complex, necessitating token-wise modeling. We are actively investigating this further — stay tuned.


## Abstract

Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. 
We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses.
Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. 
In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., 
the model might use phrases or facts in the response to question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. 
We also extend our framework and highlight a unique ‘’squeezing effect’’ to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), 
where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. 
The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.

## About this repo

The src contains the code for all experiments in the paper, but running them might be annoying. 
Sorry for not well organizing the code more systematically. 
However, as the code is based on [this repo](https://github.com/Shawn-Guo-CN/SFT_function_learning/tree/main) and the [official DPO repo](https://github.com/eric-mitchell/direct-preference-optimization), 
it is easy for you to write your own code using your favorite framework. 
The small experiments and probing dataset generation in the [notebooks](https://github.com/Joshua-Ren/Learning_dynamics_LLM/tree/main/notebook) are self-contained. 
Here is a [short document](https://weak-family-7e4.notion.site/Go-over-the-codebase-19c85a5295b8808ea13cf33f18ada4a8) for the code of this project, hope that will be helpful.

# Reference
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2407.10490) (arXiv link, would be up-to-date).
```
@inproceedings{
  ren2025learning_dynamics_LLM,
  title={Learning Dynamics of {LLM} Finetuning},
  author={Yi Ren and Danica J. Sutherland},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=tPNHOoZFl9}
}
```

# Contact
Please contact renyi.joshua@gmail.com if you have any questions about the code.


