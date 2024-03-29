# The Option-Critic Architecture

#### (Bacon PL, Harb J, Precup D)

This is a TensorFlow implementation of[] the option critic architecture](https://arxiv.org/pdf/1609.05140.pdf) by Bacon PL, Harb J and Precup D. 
The release of the code was influenced by the recent Baseline initiative from [OpenAI](https://github.com/openai/baselines) and is aimed to serve as a starting point in Hierarchical RL, more specifically in options learning.

Providing a unified common ground through the adoption of a single framework will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of as as recently pointed out by J. Schulman. This implementation of the option-critic architecture has been influenced by the original Theano implemntation by [J. Haarb](https://github.com/jeanharb/option_critic) as well as the extremely helpful repository of [D. Britz](https://github.com/dennybritz/reinforcement-learning) and the very engaging posts by [A. Juliani](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0).

We hope to be providing the community with a useful tensorflow implementation of a popular options learning architecture.

You can download it by typing:

```bash
git clone git@github.com:yadrimz/option-critic.git
```

## Requirements:
- tensorflow-1.1.0
- gym[atari]

<!--## If you are curious.

##### Train a Cartpole agent and watch it play once it converges!

Here's a list of commands to run to quickly get a working example:

<img src="data/cartpole.gif" width="25%" />


```bash
# Train model and save the results to cartpole_model.pkl
python -m baselines.deepq.experiments.train_cartpole
# Load the model saved in cartpole_model.pkl and visualize the learned policy
python -m baselines.deepq.experiments.enjoy_cartpole
```


Be sure to check out the source code of [both](baselines/deepq/experiments/train_cartpole.py) [files](baselines/deepq/experiments/enjoy_cartpole.py)!

## If you wish to apply DQN to solve a problem.

Check out our simple agented trained with one stop shop `deepq.learn` function. 

- `baselines/deepq/experiments/train_cartpole.py` - train a Cartpole agent.
- `baselines/deepq/experiments/train_pong.py` - train a Pong agent using convolutional neural networks.

In particular notice that once `deepq.learn` finishes training it returns `act` function which can be used to select actions in the environment. Once trained you can easily save it and load at later time. For both of the files listed above there are complimentary files `enjoy_cartpole.py` and `enjoy_pong.py` respectively, that load and visualize the learned policy.

##### Download a pretrained Atari agent

For some research projects it is sometimes useful to have an already trained agent handy. There's a variety of models to choose from. You can list them all by running:

```bash
python -m baselines.deepq.experiments.atari.download_model
```

Once you pick a model, you can download it and visualize the learned policy. Be sure to pass `--dueling` flag to visualization script when using dueling models.

```bash
python -m baselines.deepq.experiments.atari.download_model --blob model-atari-prior-duel-breakout-1 --model-dir /tmp/models
python -m baselines.deepq.experiments.atari.enjoy --model-dir /tmp/models/model-atari-prior-duel-breakout-1 --env Breakout --dueling
```-->
