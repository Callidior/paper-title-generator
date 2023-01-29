Paper Title Generator – Training
================================

This directory contains a python script for training a BERT2BERT model predicting paper titles from abstracts.


Prerequisites
-------------

Install the required dependencies by first following the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your platform and then running:

```bash
pip install -r requirements.txt
```

Second, you need to obtain the arXiv training dataset by downloading the file `arxiv-metadata-oai-snapshot.json` from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv). This requires a Kaggle account.


Training
--------

Run the training script as follows:

```bash
python train_title_generation.py \
    arxiv-metadata-oai-snapshot.json \
    bert-base-titlegen \
    --base-model bert-base-uncased
```

The first command-line argument is the path to the previously downloaded dataset and the second argument is the directory where the final model will be stored.
