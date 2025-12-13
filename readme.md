# This is the Github repo for the mini-project of Heinrich and Stian

Stian Lindseth
Heinrich Hegenbarth

## Main Idea
Using the concept safety neurons (SN) to predict "unsafe" inputs to an LLM.
The central problem is within the domain of human allignment of LLM. 
More specifically, it concerns how safety neurons can be identified in LLMs, 
and whether we can utilize the activations of these neurons to predict an unsafe prompt. 


## Approach

We approach this research question by downloading an aligned and an unaligned model from huggingface
We chose the `Qwen3-4b` and the `Qwen3-4b-SafeRL`. The architectures of these models are symmetric and can therefore be used to identify safety features. 
While the base model lacks the safety alignment, the SafeRL model has obtained such properties from reinforcement learning.

The safety neurons are then identified by running the same harmful prompts throught the two LLMs, and finding the activations that have the highest changes. These neurons are called safety neurons. 


## Discussion
Our results as of 02.12 is that safety neurons can be identified using inference-time activation contrasting as presented in the Chen et. al. (2025) paper. Our preliminary results also confirm the authors' conlusion that the safety neurons are sparse (the heatmap in 1_activationContrasting.py illustrates this). 
The main challenges of the project concerns the generation of the dataset used for the classification. 
This process involves caching the activations of safety neurons on a dataset of prompts which proved to be technically challenging, and computationally intensive. 

(Update 12.12.2025):
We used the safety Neurons to classify a set of safe and unsafe activations into their respective classes. Using a classifier trained sorely on safety neurons an out of sample accuracy of 96.73% could get achieved.

### Some Results

dim training: 3539 rows, 92160 activations
dim testing:  885 rows, 92160 activations
safety neurons: 4608

```
----------------safety neurons----------------
accuracy: 0.9853107344632769
recall: 0.9823008849557522
confusion matrix: [[428   5]
 [  8 444]]
classification report:               precision    recall  f1-score   support

           0       0.98      0.99      0.99       433
           1       0.99      0.98      0.99       452

    accuracy                           0.99       885
   macro avg       0.99      0.99      0.99       885
weighted avg       0.99      0.99      0.99       885
```

```
----------------pca----------------
accuracy: 0.984180790960452
recall: 0.9845132743362832
confusion matrix: [[426   7]
 [  7 445]]
classification report:               precision    recall  f1-score   support

           0       0.98      0.98      0.98       433
           1       0.98      0.98      0.98       452

    accuracy                           0.98       885
   macro avg       0.98      0.98      0.98       885
weighted avg       0.98      0.98      0.98       885
```

```
----------------full----------------
accuracy: 0.9898305084745763
recall: 0.9889380530973452
confusion matrix: [[429   4]
 [  5 447]]
classification report:               precision    recall  f1-score   support

           0       0.99      0.99      0.99       433
           1       0.99      0.99      0.99       452

    accuracy                           0.99       885
   macro avg       0.99      0.99      0.99       885
weighted avg       0.99      0.99      0.99       885
```



### Inference-time activation contrasting
![alt text](<formula_activation_contrasting.png>)


## Literature/ Reads
- Towards Understanding Safety Alignment:
A Mechanistic Perspective from Safety Neurons 
| [Chen et al., 2025](https://arxiv.org/pdf/2406.14144)

