# Keras-training-tools 

Implementation of some of the very effective tools for training Deep Learning (DL) models that I came across while doing the fastai course on [Practical Deep Learning for Coders](https://course.fast.ai/). 

The tools were first presented in the following papers by Leslie N. Smith:
- LR Finder: [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/abs/1803.09820)
- One Cycle Scheduler: [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186)

My implementations are a port of the code in [fastai library](https://github.com/fastai/fastai) (originally, based on Pytorch) to Keras and are heavily inspired by some of earlier efforts in this direction:

- https://github.com/surmenok/keras_lr_finder
- https://github.com/titu1994/keras-one-cycle

Here's another article I referred to: [How Do You Find A Good Learning Rate](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) by Sylvain Gugger of fastai which provides an intuitive understanding of how fastai's LR finder works. 

I'll keep updating this repository with the new tools I come across that could be practically useful for training a DL model.