# Gated abstention steering

A small prototype: when a language model doesn't know an answer it usually **makes one up** instead of
saying "I don't know." You can nudge it toward abstaining by adding a single direction to its
activations — but done blindly that also silences it on questions it *did* know. So this trains a cheap
linear **probe** that reads the model's own internal state to predict *"am I about to get this wrong?"*,
and fires the abstention nudge **only then**.

**Result (Gemma-2-2b, one T4 GPU, all forward passes — no fine-tuning):** blind steering drives
hallucinations down but flattens helpfulness on known questions (1.00 → 0.00). The probe-gated version
holds ~0.55 helpfulness at comparable honesty, across every steering strength.

![frontier](frontier_final.png)

*Up-and-right is better. **Red** = steer every question (blind); it gets honest but collapses helpfulness.
**Green** = the same knob, gated by the probe; it stays helpful while getting honest. The gate breaks the
trade-off.*

None of the pieces are new — diff-of-means activation steering (the primitive behind the DeepSeek-R1
feature report; CAA / Arditi) plus a correctness probe (Kadavath et al., *"Language Models (Mostly) Know
What They Know"*). The contribution is wiring the probe onto the steering knob as its trigger and
measuring the tradeoff — which the steering literature applies blind and the probing literature never
acts on. **predict → gate → apply → measure.**

## Run it

Open [`ctgt_abstention_knob.ipynb`](ctgt_abstention_knob.ipynb) in Colab or Kaggle (T4 GPU). It's
annotated top to bottom — the reasoning, the code, and the results in one place — and runs end to end in
~20–40 min. Gemma-2 is license-gated, so it needs a Hugging Face token (see the first cell).

Full limitations are in the notebook; the short version: small eval samples (read the numbers as ±~0.1),
marker-based abstention detection, and an in-distribution result rather than a worst-case guarantee.
