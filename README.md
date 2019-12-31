# Overview

A tensorflow implementation of the model proposed in "A Low Rank Weighted Graph Convolutional Approach to Weather Prediction" by Tyler Wilson, Pang-Ning Tan, and Lifeng Luo. Running the demo.py file will train and evaluate a model on the IGRA temperature prediction task described in the paper.

The implementation of the graph convolutional LSTM cell is based on Oliver Hennigh's implementation of a gridded convolutional LSTM cell available [here](https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow).

When citing, please use:
@inproceedings{wilson2018low,
  title={A Low Rank Weighted Graph Convolutional Approach to Weather Prediction},
  author={Wilson, Tyler and Tan, Pang-Ning and Luo, Lifeng},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={627--636},
  year={2018},
  organization={IEEE}
}

# Installation

To install:

1. clone the github project
2. navigate to the cloned project directory on your machine
3. create a pip virtual environment that uses python 3.5+
4. activate the pip virtual environment you just created
5. install the requirements with "pip install -r requirements.txt"
6. Run the demo with "python demo.py"
