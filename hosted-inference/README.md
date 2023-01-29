Paper Title Generator – Web Interface (Self-Hosted)
===================================================

This directory contains the source code for the inference web interface for generating paper titles from abstracts.

This version of the UI does not perform inference itself, but delegates it to the HuggingFace hosted inference API.
If you just want to try it a model locally or have a web server with sufficient compute, use the version of the interface in the `self-hosted` directory.

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
python title_generator_web.py <HUGGINGFACE-TOKEN> <MODEL-NAME> --port <PORT> --num-proc 4
```

Replace `<HUGGINGFACE-TOKEN>` with your personal access token obtained [here](https://huggingface.co/settings/tokens) and `<MODEL-NAME>` with the model hosted on HuggingFace to be used for inference (e.g., `Callidior/bert2bert-base-arxiv-titlegen`).

