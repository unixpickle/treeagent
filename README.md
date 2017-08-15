# tree-agent

In this repository, I've been experimenting with training decision trees on RL tasks.

# Branches

I've had a few different ideas for training decision trees as policies. These ideas are spread out in different branches of this repo:

 * [greedy-clone](https://github.com/unixpickle/treeagent/tree/greedy-clone) - behavior-cloning with ID3 on the better half of episodes in a batch. Solves CartPole-v0 and µniverse's Knightower-v0.
 * [dist-matching](https://github.com/unixpickle/treeagent/tree/dist-matching) - update action distributions gradually and train trees to match the new distributions. Doesn't quite solve CartPole-v0; solves µniverse's Knightower-v0. Doesn't improve on µniverse's PenguinSkip-v0.
 * [greedy-forest](https://github.com/unixpickle/treeagent/tree/greedy-forest) - build up a decision forest by adding a "greedy" tree policy at each training iteration. Doesn't quite converge on CartPole-v0; improves on µniverse's PenguinSkip-v0, but only to a mean reward of ~11 (up from ~8).
 * **this branch:** build decision forests with an algorithm akin to gradient boosting. Each tree tries to match the policy gradient estimator as well as possible. Solves CartPole-v0. *More results pending.*
