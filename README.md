[![DOI](https://zenodo.org/badge/739235639.svg)](https://zenodo.org/doi/10.5281/zenodo.10677187)
[![PyPI version](https://badge.fury.io/py/torchdecomp.svg)](https://badge.fury.io/py/torchdecomp)
![GitHub Actions](https://github.com/chiba-ai-med/PyTorchDecomp/actions/workflows/build_test_push.yml/badge.svg)


# PyTorchDecomp
A set of matrix and tensor decomposition models implemented as PyTorch classes


## Installation

Because PyTorchDecomp is a PyPI package, please install it by `pip` command as follows:

```shell
python -m venv env
pip install torchdecomp
```

For the other OS-specific or package-manager-specific installation, please check the [README.md](https://github.com/pytorch/pytorch) of PyTorch.


## Usage

```python
import torchdecomp as td
import torch


```

## References

- **Non-negative Matrix Factorization (NMF)**
  - Kimura, K. A Study on Efficient Algorithms for Nonnegative Matrix/Tensor Factorization, Ph.D. Thesis, 2017
- **Exponent term depending on Beta parameter**
  - Nakano, M. et al., Convergence-guaranteed multiplicative algorithms for nonnegative matrix factorization with Beta-divergence. IEEE MLSP, 283-288, 2010
- **Beta-divergence NMF and Backpropagation**
  - https://yoyololicon.github.io/posts/2021/02/torchnmf-algorithm/


## Contributing

If you have suggestions for how `PyTorchDecomp` could be improved, or want to report a bug, open an issue! We'd love all and any contributions.

For more, check out the [Contributing Guide](CONTRIBUTING.md).


## License

PyTorchDecomp has a MIT license, as found in the [LICENSE](LICENSE) file.


## Authors
- Koki Tsuyuzaki
