#!/bin/bash

mkdir envs && pushd envs
sudo apt-get install -y python3.9-venv
python3.9 -m venv jax
source jax/bin/activate
popd
pip install -U pip wheel
pip install jupyter jupyterlab ipython matplotlib clu tensorflow-text sentencepiece
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
git clone --branch=main https://github.com/google/flax.git
pushd flax
pip install -e .
popd
# git clone --branch=main https://github.com/levskaya/adhd.git
pushd adhd
pip install -e .
popd

# echo “set-option -g prefix C-]” > ~/.tmux.conf
