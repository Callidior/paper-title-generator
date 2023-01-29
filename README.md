Paper Title Generator
=====================

This repository contains source code for predicting computer science paper titles from abstracts using a BERT2BERT transformer model.

A demo of the inference web interface can be found at <https://paper-titles.ey.r.appspot.com/>.

Contents
--------

The repository comprises the following directories:

- [`training`](./training/): Code for training the BERT2BERT model on the arXiv dataset.
- [`self-hosted`](./self-hosted/): Code for the web interface performing inference locally.
- [`hosted-inference`](./hosted-inference/): Code for the web interface using the HuggingFace hosted inference API for inference.
