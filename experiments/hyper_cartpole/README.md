# Finding CartPole hyper-parameters

In this experiment, I try to find good hyper-parameters for training treeagent on CartPole-v0 (in OpenAI Gym). Since CartPole-v0 is a simple problem, I was able to train hundreds of models with different hyper-parameters.

This experiment had two phases. In the first phase, I recorded the results of training 6,806 models with random hyper-parameters. This data was stochastic, so it only gave me a rough idea of what hyper-parameters were ideal. In the second phase, I fit a small neural network to the results from the first phase to help reduce the effect of noise. I then used the neural network to quickly try out a bunch of different hyper-parameter configurations.

# Results

Here's a summary of what I found:

 * **Depth:** a depth of 2 or more is sufficient.
 * **Step size:** a step size of 0.2 is ideal.
 * **Truncation:** truncation usually does not help.

# Steps to reproduce

All of the models and data files in this directory can be generated with the scripts provided here. Here is what I did:

 * Generate [search_output.csv](search_output.csv): `go run main.go | tee search_output.csv`
   * Must run [gym-socket-api](https://github.com/unixpickle/gym-socket-api) in the background.
 * Generate [data](data/): `./data_gen.sh`
 * Generate *model_out*: `./train.sh` (run multiple times to train for longer)
 * Validate model on test set: `./validate.sh`
 * Try out your own hyper-parameters on the model: `./eval_params.sh`
