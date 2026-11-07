# Polyglot-Optima

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) reinforcement-learning environment that trains LLMs to rewrite Python into hardware-aware, fuzz-verified C++. The reward is measured speedup, gated by correctness, with an adversarial trap library to catch reward hacking.

- **Project README, results and design:** [`polyglot_optima/README.md`](polyglot_optima/README.md)
- **Live environment:** [Hugging Face Space](https://huggingface.co/spaces/Swastikr/polyglot_optima)
- **Write-up:** [`polyglot_optima/HF_BLOG_WRITEUP.md`](polyglot_optima/HF_BLOG_WRITEUP.md)
- **GRPO training notebook:** [`polyglot_optima/training/openenv_hackathon_training_colab_grpo_only.ipynb`](polyglot_optima/training/openenv_hackathon_training_colab_grpo_only.ipynb)

Built for the OpenEnv hackathon finale (April 2026).

## Quick start

```bash
cd polyglot_optima
pip install -e .
python -m pytest -q
```
