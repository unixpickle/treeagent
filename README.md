# tree-agent

In this repository, I've been experimenting with training decision trees on RL tasks.

# Branches

I've had a few different ideas for training decision trees as policies. These ideas are spread out in different branches of this repo:

 * [greedy-clone](https://github.com/unixpickle/treeagent/tree/greedy-clone) - behavior-cloning with ID3 on the better half of episodes in a batch. Solves CartPole-v0 and µniverse's Knightower-v0.
 * [dist-matching](https://github.com/unixpickle/treeagent/tree/dist-matching) - update action distributions gradually and train trees to match the new distributions. Doesn't quite solve CartPole-v0; solves µniverse's Knightower-v0. Doesn't improve on µniverse's PenguinSkip-v0.
 * [greedy-forest](https://github.com/unixpickle/treeagent/tree/greedy-forest) - build up a decision forest by adding a "greedy" tree policy at each training iteration. Doesn't quite converge on CartPole-v0; improves on µniverse's PenguinSkip-v0, but only to a mean reward of ~11 (up from ~8).
 * **this branch:** build decision forests with an algorithm akin to gradient boosting. Each tree tries to match the policy gradient estimator as well as possible. Solves CartPole-v0. *More results pending.*

# Log

Here is a high-level log of what I've been trying and what experiments I've run:

 * Implemented policy gradient boosting. Trees split to maximize gradient-mean dot products. Worked alright on CartPole-v0.
 * Used sum-of-gradients instead of mean-of-gradients for the tree. Made CartPole-v0 converge must faster.
 * Trained on PenguinSkip-v0 overnight: ~200 iterations, batch=512, step=20, depth=8. Got up to mean reward of ~50. Convergence at this point was very slow.
   * Tried reducing step size, increasing depth and batch size, etc. Never got past mean reward of ~53.
 * Added MSE-based algorithm to match policy gradient in the regression sense.
 * MSE-based algorithm appeared to perform worse than sum algorithm on PenguinSkip-v0 (for same step size, batch size, depth). Tried a few different step sizes; same result.
 * Added PPO-like algorithm with value function approximation.
 * Added back the original mean-gradient algorithm, and this time was able to get it to perform quite well on CartPole-v0.
 * Ran PPO overnight on PenguinSkip-v0. Used mean-gradient algorithm with depth=4, step=0.1, iters=4, batch=128. By morning, mean reward had only increased to ~13.
 * Quickly went back and tried regular policy gradient boosting (no PPO) with mean-gradients, a large step size (20), and a batch size of 128. Realized that some leaves had huge values (e.g. 35) while other leaves had tiny values (e.g. 0.001). With a step size of 20, this would have saturated the softmax function. Nonetheless, the mean entropy only seemed to decrease gradually, like I've already been observing. I realized at this point that, during all these experiments, a small fraction of samples were probably being fully saturated, while others weren't getting any help at all.
 * Implemented an RMSProp-like algorithm: turn leaf values into their sign. This should prevent any samples from being hugely saturated at once.
 * Tried the RMSProp-like algorithm combined with PPO. Ran on PenguinSkip-v0 with a step size of 0.05, a batch size of 128, and a depth of 4. It achieved a much better result by the 80th time-step than the previous PPO-like attempt that I ran overnight. However, improvement was still fairly slow and the entropy got down pretty low (~0.55).
 * Trying regular PPO + mean-gradients with a step size of 0.01. Notably, the objective improvements seem to be about the same as with a step size of 0.1. However, the value function is improving more slowly, and the entropy is still at ~1.091 after 35 batches.
 * Use the optimal step size for training value functions. Looks like that tends to cause overfitting, which a shrinkage term of ~0.25 tends to fix. Reaching an MSE of ~2.8 after 15 batches of PPO (before it was closer to ~3.4).
 * Attempted to train more stuff with PPO, gradient signs, smaller step sizes, and bigger batch sizes (e.g. 512). Step size of 0.01 was fairly unhelpful. Best model overnight still averages at ~9.
 * Looked at leaf nodes for all three algorithms. Discovered that MSE and mean-gradients are equivalent. Also discovered that the sum algorithm produces much more balanced trees.
 * In light of the above results, I started running two new PPO+sign experiments with the sum algorithm. In one experiment, I am using a big batch size (512). In the other, I'm using the default batch size, 128. In both, I lowered the `valstep` to 0.1 to prevent overfitting (which last night's experiments showed).
 * After about a day, a PenguinSkip-v0 agent got up to ~150 mean reward. Here is what was used: PPO + sign, sum algorithm, step 0.05, batch=512, iters=8, discount=0.7, depth=4, valstep=0.1.
 * Tried two different agents on Pong-v0 overnight. One had batch size 4096, one had batch size 16384. Both were at about -17 after a night of training. Trying again with a batch size of 100K but a minibatch fraction of 0.2.
 * Discovered that a step size of 0.05 with the sign algorithm was too big for the surrogate objective (loss jumping around randomly, even with full-sized mini-batches). Reduced to 0.01.
 * Run of PPO on pong with the above settings got up to a mean reward of -16.5. The entropy seemed pretty much stuck around 1.49. Reducing entropy regularization to 0.001 caused entropy to decrease down to ~1.47 pretty fast, but performance did not improve.
 * Tried implementing an algorithm with the splitting criteria of SignAlgorithm but with mean gradients. I'm hoping this might give more useful steps.
 * Played with different splitting criteria to investigate cosine distance. Realized that MSE is equivalent to maximizing cosine distance at every split. Noticed because of [this tracker I implemented](https://gist.github.com/unixpickle/2e3a4358c1c9565466198b7a410ab82f), which had behavior equivalent to MSE.
 * Implemented restriction on minimum samples in a leaf. Thus, we can force MSE to give us more balanced trees. If each leaf has enough nodes, ideally the expectation assumptions from policy gradient will hold.
 * Trained a Pong-v0 agent overnight with MSE and a step size of 0.5. By morning, reward was mid -18's and entropy was ~1.71 (higher than most other models I've trained overnight). I hypothesize that the step size I chose was too small. Trying further training with larger step sizes (e.g. 2) and larger batch sizes (50K instead of 4K).
 * Results from another day of training:
   * Pong-v0: up to ~-14.5 on the best attempt.
   * Pong-ram-v0: up to ~-14.8 on the best attempt.
   * Breakout-ram-v0: up to ~8 on best attempt. Another attempt got stuck at ~6. Maybe too big learning rate / too little regularization. On the ~6 one, paddle pretty much stays at the sides of the screen most of the time (crosses the screen fairly quickly).
 * Ran a bunch of experiments with Breakout-ram-v0.
   * Those with a learning rate of 0.1 got up to ~2.7-3.7.
   * Those with a learning rate of 0.5 got up to ~6-8 in the same amount of time.
   * Batch size was 30K. Found that minleaf=512 is way better than minleaf=128.
   * After an initial performance explosion, all models pretty much stopped learning.
   * Reducing entropy didn't help much, nor did decreasing learning rate.
 * Conclusion: after the first ~80 batches, improvement always stops, regardless of learning rate.
   * Hypothesis: forest size is the cause. Huge forests might be hard to improve?
   * Plan: implement tree decay and pruning. Can keep the forest fairly small. Will it work? Idk.
 * New theory: the value function needs pruning, but not necessarily the policy. Deleting the critic.json file after a while of training speeds up training that seems to have stalled.
 * Training overnight on Breakout-ram-v0 with a value function decay rate of 0.99 has resulted in a mean reward of ~16, nearly twice as much as any previous attempts. Training has been slow since about ~13, but it is still ongoing. Mean entropy is down to about 1.11.
 * Me an hour ago: ugh, why is this thing so bad at MuJoCo. Me now: oh, uint8 compression.
