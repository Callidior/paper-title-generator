Paper Title Generator – Gradio App
==================================

This directory contains the source code for the [Gradio](https://gradio.app/) app for generating paper titles from abstracts, which is running as a Huggingface Space: <https://huggingface.co/spaces/Callidior/arxiv-titlegen>

If a GPU is available, this app will load the model and perform inference itself, otherwise it will delegate requests to the Huggingface Hosted Inference API.

Prerequisites
-------------

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Launch
------

Spin up the web interface by running:

```bash
python app.py
```
