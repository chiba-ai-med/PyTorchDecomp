[![DOI](https://zenodo.org/badge/739235639.svg)](https://zenodo.org/doi/10.5281/zenodo.10677187)
[![PyPI version](https://badge.fury.io/py/torchdecomp.svg)](https://badge.fury.io/py/torchdecomp)
![GitHub Actions](https://github.com/chiba-ai-med/PyTorchDecomp/actions/workflows/build_test_push.yml/badge.svg)


# PyTorchDecomp
A set of matrix decomposition algorithms implemented as PyTorch classes


## Installation

Because PyTorchDecomp is a PyPI package, please install it by `pip` command as follows:

```shell
python -m venv env
pip install torchdecomp
```

For the other OS-specific or package-manager-specific installation, please check the [README.md](https://github.com/pytorch/pytorch) of PyTorch.


## Usage

See the [tutorials](https://chiba-ai-med.github.io/PyTorchDecomp/tutorials.html).

## References

- **LU/QR/Cholesky/Eigenvalue Decomposition**
  - Gene H. Golub, Charles F. Van Loan Matrix Computations (Johns Hopkins Studies in the Mathematical Sciences)
- **Principal Component Analysis (PCA) / Partial Least Squares (PLS)**
  - R. Arora, A. Cotter, K. Livescu and N. Srebro, Stochastic optimization for PCA and PLS, 2012 50th Annual Allerton Conference on Communication, Control, and Computing, 2012, 861-868. 2012
- **Independent Component Analysis (ICA)**
  - Hybarinen, A. and Oja, E. Independent component analysis: algorithms and applications, Neural Networks, 13, 411-430. 2000
- **Deep Deterministic ICA (DDICA)**
  - H. Li, S. Yu and J. C. Príncipe, Deep Deterministic Independent Component Analysis for Hyperspectral Unmixing, 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 3878-3882, 2022
- **Non-negative Matrix Factorization (NMF)**
  - Kimura, K. A Study on Efficient Algorithms for Nonnegative Matrix/Tensor Factorization, Ph.D. Thesis, 2017
  - **Exponent term depending on Beta parameter**
    - Nakano, M. et al., Convergence-guaranteed multiplicative algorithms for nonnegative matrix factorization with Beta-divergence. IEEE MLSP, 283-288, 2010
  - **Beta-divergence NMF and Backpropagation**
    - https://yoyololicon.github.io/posts/2021/02/torchnmf-algorithm/

## Contributing

If you have suggestions for how `PyTorchDecomp` could be improved, or want to report a bug, open an issue! We'd love all and any contributions.

For more, check out the [Contributing Guide](https://github.com/chiba-ai-med/PyTorchDecomp/blob/main/CONTRIBUTING.md).


## License

PyTorchDecomp has a MIT license, as found in the [LICENSE](https://github.com/chiba-ai-med/PyTorchDecomp/blob/main/LICENSE) file.


## Authors
- Koki Tsuyuzaki
