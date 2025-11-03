# LLM Post-Training

Post-training methods
* Fine-tuning: SFT
* Reinforcement Learning: RL (PPO, GRPO)

# SFT
## LoRA Adapter
smaller number of weights can be applied to the model to adapt to different tasks. The training of LoRA adapter requires less compute and less data.

## Error Analysis
Prepare dataset, train on the dataset, and see where it doesn't work and cause error. We call that "error analysis".

"Discipline loop":
* Train model
* Evaluate model
* Fix the training data
* efficiently repeat this loop

Error Analysis is still a human skill to find the gap in the LLM and improve the LLM

# Reinforcement Learing
Underlines agentic behavior such as reasoning behavior

Rewards and Reward Functions.
In case of reasoning model, it may take many steps of reasoning to arrive at a correct conclusion. We don't want to specifiy the one way to reason. It turns out RL allows you to specify reward function to measure whether or not the final output is correct.

Let the RL algorithm try lots of different reasoning traces, and measure whether or not it gets the correct final output.

Effective to train reasoning models, and general system like computer use model.

RL can enable superhuman capabilities which makes it very attractive, but RL is also unstable today.

Stable model means, it can run more steps and the model doesn't collapse.

## PPO

## GRPO: Group Relative Policy Optimization
GRPO Algorithm - Group Relative Policy Optimization (DeepSeek): 
efficient RL learning, allows multiple rollouts, multiple attempts within the group of attempts. Figuring out which attempts worked and use that one to automatically score or give reward signal to let the algorithm fine-tune the model.

# Post-training Goal
Post-training helps you to steer the model towards your business direction and your business needs.

## Reference
* [Fine-tuning & RL for LLMs: Intro to Post-training](https://learn.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training/lesson/pu5tb0/key-components-to-making-fine-tuning-and-rl-work)


