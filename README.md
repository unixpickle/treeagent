# tree-agent

In this repository, I've been experimenting with training decision trees on RL tasks.

# Branches

I've had a few different ideas for training decision trees as policies. These ideas are spread out in different branches of this repo:

 * [greedy-clone](https://github.com/unixpickle/treeagent/tree/greedy-clone) - behavior-cloning with ID3 on the better half of episodes in a batch. Solves CartPole-v0 and µniverse's Knightower-v0.
 * [dist-matching](https://github.com/unixpickle/treeagent/tree/dist-matching) - update action distributions gradually and train trees to match the new distributions. Doesn't quite solve CartPole-v0; solves µniverse's Knightower-v0. Doesn't improve on µniverse's PenguinSkip-v0.
 * **this branch** - build up a decision forest by adding a "greedy" tree policy at each training iteration. *Results pending.*
