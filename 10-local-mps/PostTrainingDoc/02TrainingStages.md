# LLM Training Stages

## Pre-Training
Model learns to generate text vias next token prediction.
It starts with untrained model with completely randomized weights It can take moths of compute to get that pre-trained base model. Throught the next token prediction, with a huge amount of data, the base model is able to learn concepts.

## Mid-Training
Usually with a different team in the frontier lab. 
Continuous pre-training, with a more specific curated dataset. It is still predicting the next token, but essentially this phase is used to target e.g. new languages.

It can also be a good place to add different modalities, such as audio or images.

It is also been used to increase the context length for the model.

## Post-Training
Uses method such as 
**fine-tuning (SFT)**: you give the model different inputs, and also target outputs of exactly how model should respond to a particular input.

Output gradients, efficient fine-tuning with LoRA adapter.

**Reinforcement learning**: which teaches the model based on a certain input, whether its response was good or bad. It gives reward or score for the model response.

Get the reward or score using another model, and train the reward model on human preferences in reinforcement learning with human feedback.

4 different models for RLHF, and computationally expensive.

## Insights
Compare:
* Base model
* finetuned model
* reinforcement learned model
with an example prompt and how they behave very different.

Pre-training: just reading the data content
Mid-training: data are little bit more curated, the model reads a curated set and be able to learn either new languages or new domains, but still reading a huge amount of books.
Post-training: SFT, RLHF, the model learns to format.


# Reference
* https://sam-wei.medium.com/a-practical-breakdown-of-training-pre-training-post-training-and-fine-tuning-e39b95db257d