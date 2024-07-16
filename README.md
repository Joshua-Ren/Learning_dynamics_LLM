# Learning_dynamics_LLM

This repository contains the official code for the paper "[Learning Dynamics of LLM Finetuning]()".

Authors: [Yi Ren](https://joshua-ren.github.io/), and [Danica J. Sutherland](https://djsutherland.ml/).

## Abstract
Learning dynamics, which describes how the learning of specific training examples influences the model's prediction of other examples, 
give us a powerful tool for understanding the behavior of deep learning systems.
We study the learning dynamics of large language models during finetuning, 
by analyzing the step-wise decomposition and accumulated influence among different responses.
Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. 
The analysis not only explains where the benefits of these methods come from 
but also inspires a simple, effective method to further improve the alignment performance.

## Interesting findings

- Understanding the model's behavior using learning dynamics provides insights from another perspective than studying the converged behavior;
- During both SFT and DPO, the behavior of different responses matches our analysis well;
- The **squeezing effect** when gradient ascent exists is quite counter-intuitive and can explain many special behaviors of DPO;

## About this repo
The src contains the code for all experiments in the paper, but running them might be annoying. Sorry for not well organizing the code more systematically. However, as the code is based on [this repo](https://github.com/Shawn-Guo-CN/SFT_function_learning/tree/main) and the [official DPO repo](https://github.com/eric-mitchell/direct-preference-optimization), it is easy for you to write your own code using your favorite framework. The small experiments and probing dataset generation in the [notebook](https://github.com/Joshua-Ren/Learning_dynamics_LLM/tree/main/notebook) are self-contained.

## Reference
For technical details and full experimental results, please check [our paper]().
```
@misc{ren:llm_LD,
  title={Learning Dynamics of LLM Finetuning},
  author={Yi Ren and Danica J. Sutherland},
  eprinttype={arXiv},
  eprint={},
  year={2024}
}
```

## Contact
Please contact renyi.joshua@gmail.com if you have any questions about the codes.

