# Finding CartPole hyper-parameters

In this experiment, I find good hyper-parameters for training treeagent on CartPole-v0 (in OpenAI Gym). Since CartPole-v0 is a simple problem, I was able to train hundreds of models with different hyper-parameters.

This experiment had two phases. In the first phase, I recorded the results of training 735 models with random hyper-parameters. This data was stochastic, so it only gave me a rough idea of what hyper-parameters were ideal. In the second phase, I fit a small neural network to the results from the first phase to help reduce the effect of noise. I then used the neural network to quickly try out a bunch of different hyper-parameter configurations.

# Results

From the above analysis, I found the following:

 * **Step size:** close to 1 is ideal.
 * **Entropy regularization:** not needed and can hurt the model.
 * **Depth:** 3 is fine; adding more depth doesn't help much.
 * **Decay rate:** 1 (no decay) is fine.

# Steps to reproduce

All of the models and data files in this directory can be generated with the scripts provided here. Here is what I did:

 * Generate [search_output.csv](search_output.csv): `go run main.go | tee search_output.csv`
   * Must run [gym-socket-api](https://github.com/unixpickle/gym-socket-api) in the background.
 * Generate [data](data/): `./data_gen.sh`
 * Generate *model_out*: `./train.sh` (run multiple times to train for longer)
 * Validate model on test set: `./validate.sh`
 * Try out your own hyper-parameters on the model: `./eval_params.sh`
