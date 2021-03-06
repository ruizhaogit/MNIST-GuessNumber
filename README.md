# MNIST-GuessNumber Dataset

<p align="center">
  <img width="600" height="" src="https://github.com/ruizhaogit/MNIST-GuessNumber/blob/master/guess_number.png">
</p>

In the MNIST GuessNumber dataset, each sample consists of an image (left), a set of sequential questions with answers (right), and a target digit. The goal of this game is to find out the target digit through a multi-round question-answering.

## Introduction:  

This repository is based on Python3 and generates the MNIST-GuessNumber dataset.  

The code was developed by Rui Zhao (Siemens AG & Ludwig Maximilian University of Munich).  

The creation of MNIST-GuessNumber dataset is inspired by [MNIST-Dialog dataset][1].

MNIST-GuessNumber dataset is a lightweight goal-oriented visual dialog testbed.

It is for quick test of Reinforcement Learning (RL) algorithms in dialog settings.

We designed this MNIST guess-number game and used it in our paper "Efficient Dialog Policy Learning via Positive Memory Retention".

The paper is published in 2018 IEEE Spoken Language Technology (SLT), link: https://ieeexplore.ieee.org/document/8639617.

The preprint version of the paper is avaliable at: https://arxiv.org/abs/1810.01371

This work was also presented at NIPS 2018 Visually Grounded Interaction and Language (ViGIL) Workshop.



## Usage:  

```
python generate_dataset.py
```

Then you can find the generated images and dialogs in the data folder :)

## Citation:

Citation of the paper:

```
@inproceedings{zhao2018efficient,
  title={Efficient Dialog Policy Learning via Positive Memory Retention},
  author={Zhao, Rui and Tresp, Volker},
  booktitle={2018 IEEE Spoken Language Technology Workshop (SLT)},
  pages={823--830},
  year={2018},
  organization={IEEE}
}
```

## Licence:

MIT


[1]: http://cvlab.postech.ac.kr/research/attmem/

