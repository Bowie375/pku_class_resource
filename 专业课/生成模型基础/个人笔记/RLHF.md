## Reinforcement Learning from Human Feedback

Q: why do we need this?
A: Language models trained from next-token-prediction are not aligned with user intent

    Prompt        Explain the moon landing to a 6 year old in a few sentences.
    Completion    GPT-3
                  Explain the theory of gravity to a 6 year old
                  Explain the theory of relativity to a 6 year old
                  Explain the big bang theory to a 6 year old
                  Explain evolution to a 6 year old

Reason: there are not enough conversation or command-like data used in training, as they are hard to collect. So model tends to drift from expected output.

## The pretraining and then finetuning paradigm

Empirically, we cannot say for sure what we should learn and what we can learn during pretraining or finetuning, but there are some common ideas that people holds:

1. What can be learnt by model during pretraining:
    - Setting: train on lots of text, learn general things
    - For language model: 
        - grammar: eg. a verb is not likely to follow another verb, unless there is a "and" between them
        - semantic
        - knowledge

2. What can be learnt by model during finetuning:
    - Setting: use not many labels, adapt to specific task
    - Effect: improve the model by aligning the human's need

3. What exactly does human need:
    - Good format: solve tasks through conversation/instruction...
        - Method: instruction finetuning

    - Value: answer questions correctly, tailor to human preference (shouldn't teach student to cheat in exam)
        - Method: RLHF

## Instruction finetuning
- Collect examples of (instruction, output) pairs across many tasks
- Finetune an LM using next token prediction loss
- Evaluation

## RLHF
![](assets/RLHF1.png)
![](assets/RLHF2.png)
![](assets/RLHF3.png)

## Limitations of RL+Reward Modeling
- Human preferences are unreliable!
- "Reward hacking" is a common problem in RL
    - model can find shortcut solutions to fool the reward function
- Chatbots are rewarded to produce responses that seem authoritative and helpful, regardless of truth
    - This can result in making up facts + hallucinations

## Direct Preference Optimization (DPO)

DPO is another approach to RL+Reward modeling. The biggest difference is that it doesn't optimize the policy, it directly finds the optimal policy in theory.

![](assets/RLHF4.png)
![](assets/RLHF5.png)
![](assets/RLHF6.png)

## RL with Veriable Reward
- Many tasks are verifiable tasks
    - Math Problems
    - Code Problems

- From Model-based reward model to Rule-based Reward model

- Limitations:
    - Reward hacking: models exploit verifier loopholes, templates, or test case, instead of truly solving the task.
    - Cost & latency: unit tests, sandboxes, and search tasks are expensive and slow.

    - Metric bias: relying on pass@1 alone can mislead; pair with pass@k and process-quality metrics.
