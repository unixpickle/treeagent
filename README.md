# tree-agent

In this small experiment, I apply decision trees (and random forests) to reinforcement learning.

# How it works

It is not obvious how to apply decision trees to RL, especially if you want to use algorithms like [ID3](https://en.wikipedia.org/wiki/ID3_algorithm). Decision trees are classifiers, so value function methods don't make much sense. Moreover, since standard decision trees are not differentiable, there is no clear way to use policy gradient methods.

Since decision trees are classifiers, it *is* possible to train them to clone behavior. In other words, you can train a decision tree to imitate demonstrations from a good policy (e.g. a human). The key insight is that you can get good demonstrations by running a policy a ton of times and selecting the rollouts with the best reward. This results in a hill-climbing algorithm.

In the hill-climbing algorithm implemented in [trainer.go](trainer.go), a decision tree or random forest is used to implement a stochastic policy. After a lot of environment rollouts, the rollouts/actions are sorted and the better half is selected. A new tree/forest is then trained to clone the behavior of the old policy on the selected (good) actions. This process is repeated iteratively.

# Motivation

One motivation for decision trees is interpretability. If you train a decision tree agent, you will likely be able to understand how and why it works.

Another motivation is more general: if you can get RL to work well with decision trees, maybe you can get it to work with *any* kind of classifier.

# Results

The results are not very impressive, at least right now. I've managed to get an agent to solve CartPole-v0, but it takes about ~1500 environment rollouts. I also succeeded at using a decision tree policy on a simple pixel-based video game. However, I also failed to train tree-based agents on a handful of other games.
