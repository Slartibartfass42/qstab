# QStab

## A Python Framework For the Simulation and Analysis of Random Quantum Clifford Circuits and Stabilizer Codes.

Used to generate initial data about the NoRA tensor network ansatz as presented in [arXiv:2303.16946](https://arxiv.org/abs/2303.16946).

**Disclaimer:** Work on this implementation of QStab has stopped due to the inefficiency of the Python language for large-scale data generation. Please refer to [QStab.jl](https://github.com/vbettaque/QStab.jl) for the currently maintained implementation using Julia.

## Main Dependencies

- [**galois** (v0.1.1)](https://github.com/mhostetter/galois)

- [**numpy** (v1.21.6)](https://github.com/numpy/numpy)

Newer versions might also work, but have not been tested.

## Usage

The code relevant for the data generation in [arXiv:2303.16946](https://arxiv.org/abs/2303.16946) is primarily found in `syk.py`, whereas new random Cliffords/Weyls and stabilizers can be generally sampled using the code in `clifford.py` and `stabilizer.py` respectively. All important functions are commented indicating how to use them.

Most functions also require a Galois field object (`GF`) as input parameter, which indicates the qudit dimension and can be generated using `galois.GF(p**m)` for some prime number `p > 2` and positive integer `m > 0`. For the paper `p = 3` and `m = 1` were used.