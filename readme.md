# This is the Github repo for the miniproject in AML4NLP

## Main Idea
Using the concept safety neurons (SN) to predict "unsafe" inputs to an LLM

### Safety Neurons
SN are the neurons that activate if an "unsafe" input is passed to an LLM

## Literature/ Reads
- Toy Models of Superposition 
| [Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html)
- Foundational Challenges in Assuring Alignment and
Safety of Large Language Models 
| [Anwar et al., 2024](https://arxiv.org/pdf/2404.09932) (2.2, 2.7, 3.1, 3.2)
- Assessing the Brittleness of Safety Alignment
via Pruning and Low-Rank Modifications 
| [Wei et al., 2024](https://arxiv.org/pdf/2402.05162)
- Towards Understanding Safety Alignment:
A Mechanistic Perspective from Safety Neurons 
| [Chen et al., 2025](https://arxiv.org/pdf/2406.14144)

### Githubs
- implementation chen paper: https://github.com/THU-KEG/SafetyNeuron.git

## Approach
1. Compare approaches from Literature
2. Find Small LLM and run Locally 
3. Use dataset with unsafe prompts to identify SN
    3.1. Visualize SN -> validate their existance
4. Train classifier on them to check output.
5. Scale up to larger LLM using HPC


## Which LLM to use. 


## LLama github repo
https://github.com/meta-llama


