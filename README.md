ADHD
======

Attention modeling for those without it.

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
