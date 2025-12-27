### Disadvantages of n-gram LM
Given sentence length n, three metrics: sparcity, memory and accuracy cannot be achieved at the same time.
    
- when n decreases:
    - less ocntext + worse accuracy
    - less memory consumption
    - statistically better estimations

- when n increases:
    - more ocntext + better accuracy
    - more memory consumption
    - statistically worse estimations

__Q:__ how to understand "sparcity" and "accuracy" here?
    Accuracy means the approximation of the true possibility of $$ \frac{P(X_n, X_{n-1},..., X_{n-k})}{P(X_{n-1}, X_{n-2},..., X_{n-k})} $$ is better becauses we got more samples.
    Sparcity means the number of word compositions that occurs only once or twice becomes more and more. These compositions will lead to worse performance. 


### What leads to the decision of the loss function of RNN?
For Autoregressive language model, we want to maximize the probability of the sentences in corpus under other parametrized model $P_\theta(X_{n}|X_{n-1}, X_{n-2},..., X_{1})$. So we want to maximize $$ P(X_{n}, X_{n-1}, \ldots , X_{1}) $$ which can be factorized into $$ P(X_{n}|X_{n-1}, X_{n-2}, \ldots, X_{1})P(X_{n-1}|X_{n-2}, \ldots, X_{1}) \cdots P(X_{1}) $$ Maximizing this can be writen into minimizing the negative log-likelihood which is the loss function of RNN.

### Training and inference efficiency of RNN and Transformer
- Training:
    - RNN: O(N) computation complexity, N is sentence length
    - Transformer: O(N^2) computation complexity, N is sentence length. But highly optimized and parallelized.
    - Transformer is more efficient overall.

- Inference:
    - RNN: O(N) computation complexity, N is sentence length
    - Transformer: O(N^3) computation complexity, N is sentence length.
    