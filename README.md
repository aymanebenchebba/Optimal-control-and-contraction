# Optimal-control-and-contraction
This repository contains the Matlab code and the figure associated in order to perform the numerical applicatio presented in the paper 'Optimal controls with incremental ISS guarantees for systems with globally Lipschitz nonlinearities''

## Contents

* `contraction_verification.m` — MATLAB script for the contraction verification.
* `state_trajectories.png` — simulation result showing the evolution of the error trajectory.

## Requirements

* MATLAB
* [YALMIP](https://yalmip.github.io/)
* An SDP solver compatible with YALMIP

## Usage

Run the following script in MATLAB:

```matlab
contraction_verification.m
```

The script solves the corresponding LMI and generates the numerical results presented in the article.
