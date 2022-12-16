ADHD
======

Attention modeling for those without it.

Install
========
manual setup:

```
mkdir envs && pushd envs
sudo apt install python3.9-venv
python3.9 -m venv jax
source jax/bin/activate
popd
pip install -U pip wheel
pip install jupyter jupyterlab ipython matplotlib clu tensorflow-text
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# need recent flax
git clone --branch=main https://github.com/google/flax.git
pushd flax
pip install -e .
popd
git clone --branch=main https://github.com/levskaya/adhd.git
pushd adhd
pip install -e .
popd
```

Status
======

- Really basic training loop on LM1B and inline inference / decoding "works".
- Absolutely nothing is tuned.

TODO
====

 - What decoder model variant do we actually want?
 - Multihost data-loading and training (from sholto's library.)
 - More flexible demo prompting / simple batch inference script.
 - Prefix-LM support for input->target datasets.
 - Should we use CLU metric helpers or hand-roll that stuff?
 - We have simple tf.data pipeline, but should we use SeqIO? Grain? an outside library?
