#!/bin/bash
sudo apt-get install -y python3.9-venv
python3.9 -m venv py39
source py39/bin/activate
pip install -U pip wheel
pip install jupyter jupyterlab ipython matplotlib clu tensorflow-text sentencepiece
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
git clone --branch=main https://github.com/google/flax.git
pip install -e flax
pip install -e adhd
export TFDS_DATA_DIR="gs://tensorflow-datasets/datasets"
