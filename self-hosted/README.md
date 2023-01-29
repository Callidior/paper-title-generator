Paper Title Generator – Web Interface (Self-Hosted)
===================================================

This directory contains the source code for the inference web interface for generating paper titles from abstracts.

This version of the UI will load the model itself and use it directly. If you are instead looking for a version of the UI delegating inference to HuggingFace, see the `hosted-inference` directory.

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
python title_generator_web.py <path-to-model-dir> --tokenizer <path-to-model-dir>
```
