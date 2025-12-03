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


## Discussion
Our results as of 02.12 is that safety neurons can be identified using inference-time activation contrasting
as presented in the Chen et. al. (2025) paper. Our preliminary results also confirm the authors' conlusion
that the safety neurons are sparse (the heatmap in 1_activationContrasting.py illustrates this). 
The main challenges of the project concerns the generation of the dataset used for the classification. 
This process involves caching the activations of safety neurons on a dataset of prompts which proved to 
be technically challenging, and computationally intensive. 


### Inference-time activation contrasting
![alt text](<WhatsApp Image 2025-12-02 at 16.28.35.jpeg>)


## Literature/ Reads
<<<<<<< HEAD
- Towards Understanding Safety Alignment:
=======
>>>>>>> 4551cd7df4304754427508433dc9751ab6418f52
A Mechanistic Perspective from Safety Neurons 
| [Chen et al., 2025](https://arxiv.org/pdf/2406.14144)

