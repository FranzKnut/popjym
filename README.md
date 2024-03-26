<p align="center">
    <a href="docs/images/popjym.png">
        <img src="docs/images/popjym.png" alt="POPJym logo" width="30%"/>
    </a>
</p>

# POPJym: Partially Observable Process Gym in JAX

POPJym is POPGym in JAX. Original POPGym Paper can be found [here](https://openreview.net/forum?id=chDrutUTs0K). The Structured State Space Models for In-Context Reinforcement Learning paper found [here](https://arxiv.org/abs/2303.03982). **I have completely and utterly stolen the code from** https://github.com/luchris429/popjaxrl and slightly cleaned and formatted their code for my own personal use, truly all credit goes to the original authors.

## Quickstart Install

```python
pip install popjym
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

For e.g.
```
pip install "jax[cuda12_pip]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quickstart Usage

```python
import jax
import popjym
seed = jax.random.PRNGKey(0)
env, env_params = popjym.make(env_name)

env.reset(seed, env_params)

env.step(seed, state, action)
```

# Contributing
Please follow the coding style by using pre-commit.

```python
pip install pre-commit
pre-commit install
```

# Citing

If used in your work, please cite **a)** the original POPGym paper and **b)** the Structured State Space Models for In-Context Reinforcement Learning paper:
```
@inproceedings{
morad2023popgym,
title={{POPG}ym: Benchmarking Partially Observable Reinforcement Learning},
author={Steven Morad and Ryan Kortvelesy and Matteo Bettini and Stephan Liwicki and Amanda Prorok},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=chDrutUTs0K}
}
```
```
@article{lu2023structured,
  title={Structured State Space Models for In-Context Reinforcement Learning},
  author={Lu, Chris and Schroecker, Yannick and Gu, Albert and Parisotto, Emilio and Foerster, Jakob and Singh, Satinder and Behbahani, Feryal},
  journal={arXiv preprint arXiv:2303.03982},
  year={2023}
}
```
