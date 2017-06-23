# tree-agent

This is an experiment to see how well random forests of decision trees can do on reinforcement learning tasks.

# How it works

It is not obvious how to apply decision trees to RL. Decision trees are classifiers, so value function methods don't make much sense. Moreover, since decision trees are not differentiable, there's no clear way to use policy gradient methods.

Instead of value function approximation or policy gradients, tree-agent uses a hill-climbing algorithm. The forest is used to implement a stochastic policy. After a lot of environment rollouts, rollouts are sorted and the better half is selected. A new forest is then trained to clone the behavior of the old forest on the selected rollouts.
