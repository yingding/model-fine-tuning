## Key Components for SFT and RLHF

### SFT Data
Dataset examples include chat history, with prompt tags

```python
{
    input="""
        <user>what's the captial of France?</user>
        <assistant>paris</assistant>
        <user>what about spain?</user>
    """,
    target_output="Madrid"
}
```


rational think tags
```python
{
    input="""
        <user>Alice has 3 apples and buys 2 more. How many now?</user>
    """,
    target_output="""<think>Start with 3. Buys 2 => 3+2=5.</think>
    <answer>5</answer>
    """
}
```

### RL: grading

#### RL: Deterministic graders - Verifiers
```python
{
    input="""
        <user>Carly has 8 apples and buys 2 more, but then sells 5 to the local baker. How many now?</user>
    """,
    model_output="""<think>8+2-5=5</think>
    <answer>0</answer>
    """
    grader="-1" # Incorrect -1
}
```

or show score

```python
{
    input="""
        <user>Carly has 8 apples and buys 2 more, but then sells 5 to the local baker. How many now?</user>
    """,
    model_output="""<think>8+2-5=5</think>
    <answer>0</answer>
    """
    grader="""incorrect:-1
    shows works: +1
    total reward (score): 0
    """
}
```
If the model's output was incorrect, your grader can give some partial credit. 

higher total score
```python
{
    input="""
        <user>Carly has 8 apples and buys 2 more, but then sells 5 to the local baker. How many now?</user>
    """,
    model_output="""<think>8+2-5=5</think>
    <answer>5</answer>
    """
    grader="""correct:+1
    shows works: +1
    total reward (score): +2
    """
}
```

Have a grader include a formatting grader
```python
{
    input="""
        <user>Carly has 8 apples and buys 2 more, but then sells 5 to the local baker. How many now?</user>
    """,
    model_output="""<think>8+2-5=5</think>
    <answer>5</answer>
    """
    grader="""correct:+1
    shows works: +1
    answer in <answer> tags: +1
    total reward (score): +3
    """
}
```
to encourage the model to produce tags correctly.

#### RL: LLM grader

```python
{
    input="""
        <user>Carly has 8 apples and buys 2 more, but then sells 5 to the local baker. How does Carly feel?</user>
    """,
    grader="""LLM or another model to output reward (score)
    """
}
```
You can learn how to grade an output of "How does Carly feel" with a score base don different parameters like politeness or enthusiasm or engagement.


```python
{
    input="""
        <user>Greet politely</user>
    """,
    model_output="""<assistant>Hi there! How are you?</assistant>
    """
    grader="""High score on politeness: +1
    High score on enthusiams: +1
    High score on engagement: +1
    Total reward (score): +3
    """
}
```

#### RL: Reward hacking
```python
{
    input="""
        <user>Greet politely</user>
    """,
    model_output="""<assistant>Hello, hello, hello, hello, hello, hello!!!!</assistant>
    """
    grader="""High score on politeness: +1
    High score on enthusiams: +1
    High score on engagement: +1
    Total reward (score): +3
    """
}
```
A common thing that happens in reinforcement learning called reward hacking.
If the grader is not fully right, the model will find a way to hack the grader.

#### RL: Input distribution

```python
{
    inputs=[
        "<user>Greet politely</user>",
        "<user>Greeet politelyy</user>",
        "<user>Hi hi Hi</user>",
    ]
}.  
```
Distribution of inputs matters a lot for training the model effectively.

In Reinforcement learning, you wan tto give that large distribution of inputs similar to that in fine-tuning., The model can react to many different types of possible user inputs.

This need to be very representative of what kind of user inputs, you expect the model to see.

#### RL: training environment
RL training environment: inputs + graders + other things

```python
{
    input="""
        <user>2+3?</user>
    """,
    model_output="""<assistant>Using the caculator_tool, 2+3=5</assistant>
    """
    grader="""Correct: +1
    Total reward (score): +1
    """
}
```
E.g. in the training environment, you may expect model to use a `calculator_tool`, or a search api and file tool.

How representative this training environment is to your real-world use case is going to be as much information and as helpful as you need the model to learn from this environment.
The more representative this training environment is to where you expect the model to operate, the better.

#### RL: data
different as data for SFT.

```python
{
    input="""
        <user>Debug this issue for me...</user>
    """,
    model_output="""<assistant>Looking at my_codebase, and getting the latest with the search_api ...</assistant>
    """
    grader="""Correct: +1
    ...
    Total reward (score): +2
    """
}
```

RL Data: `{input, output, reward}`

#### RL: training environments

Multiple training environments: a weighted mixture of their data for model to learn from different training environment.

Multiple training environment, producing multiple types of data, to train your model.

## Recap
1. Get fine-tuning data `{input, target_output}`
2. Fine-tune LLM -> fine-tuned LLM
3. Create RL training environment with different `{input}` distribution data, graders, other info (files, tools, etc)

4. RL Loop:
    a. Get RL data `{input, output, reward}` in RL training environments
    b. Train fine-tuned LLM with RL








