# This is the Github repo for the mini-project of Heinrich and Stian

Heinrich Hegenbarth
Stian Lindseth

## Main Idea
Using the concept safety neurons (SN) to predict "unsafe" inputs to an LLM.
The central problem is within the domain of human allignment of LLM. 
More specifically, it concerns how safety neurons can be identified in LLMs, 
and whether we can utilize the activations of these neurons to predict an unsafe prompt. 


## Approach

We approach this research question by downloading an aligned and an unaligned model from huggingface
We chose the Qwen3-4b and the Qwen3-4b-SafeRL. These models work well for this research as they have the same architecture, 
but the base model lacks the safety alignment obtained from reinforcement learning, which the aligned model has.

The safety neurons are then identified by running the same harmful prompts throught the two LLMs, and finding the activations
that changed the most. These are our safety neurons. 


### Safety Neurons
SN are the neurons that activate if an "unsafe" input is passed to an LLM

## Literature/ Reads
- Towards Understanding Safety Alignment:
A Mechanistic Perspective from Safety Neurons 
| [Chen et al., 2025](https://arxiv.org/pdf/2406.14144)

