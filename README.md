# POPJym: Partially Observable Process Gym in JAX

POPJym is POPGym in JAX. The code has been taken from https://github.com/luchris429/popjaxrl. Original POPGym Paper can be found [here](https://openreview.net/forum?id=chDrutUTs0K). POPJym was originally created for the Structured State Space Models for In-Context Reinforcement Learning paper found [here](The https://arxiv.org/abs/2303.03982). I have currently stolen and simply formatted their code for my own personal use, all credit goes to the original authors.

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

print(env.reset(seed, env_params))
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

## References and Acknowledgments

The code implementations here are heavily inspired by:

- [POPGym](https://github.com/proroklab/popgym)
- [S5](https://github.com/lindermanlab/S5/tree/main)
- [Gymnax](https://github.com/RobertTLange/gymnax)
- [PureJaxRL](https://github.com/luchris429/purejaxrl/tree/main)

If you use the relevant components from above, please also cite them. This includes:

S5
```
@inproceedings{
smith2023simplified,
title={Simplified State Space Layers for Sequence Modeling},
author={Jimmy T.H. Smith and Andrew Warrington and Scott Linderman},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Ai8Hw3AXqks}
}
```

Gymnax
```
@software{gymnax2022github,
  author = {Robert Tjarko Lange},
  title = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
  url = {http://github.com/RobertTLange/gymnax},
  version = {0.0.4},
  year = {2022},
}
```

PureJaxRL
```
@article{lu2022discovered,
    title={Discovered policy optimisation},
    author={Lu, Chris and Kuba, Jakub and Letcher, Alistair and Metz, Luke and Schroeder de Witt, Christian and Foerster, Jakob},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    pages={16455--16468},
    year={2022}
}
```
