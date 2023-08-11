# A Novel Framework for Optimizing Gurney Flaps using RBF Neural Network and Cuckoo Search Algorithm

Enhancing aerodynamic efficiency is vital for optimizing aircraft performance and operational effectiveness. It enables greater speeds and reduced fuel consumption, leading to lower operating costs. Hence, the implementation of Gurney flaps represents a promising avenue for improving airfoil aerodynamics. The optimization of Gurney flaps holds considerable ramifications for improving the lift and stall characteristics of airfoils in aircraft and wind turbine blade designs. The efficacy of implementing Gurney flaps hinges significantly on its design parameters, namely, flap height and mounting angle. This study attempts to optimize these parameters using a design optimization framework, which incorporates training a Radial Basis Function surrogate model based on CFD data from two-dimensional (2D) Reynolds-Averaged Navier-Stokes (RANS) simulations. The Cuckoo Search algorithm is then employed to obtain the optimal design parameters and compared with other competing optimization algorithms. The optimized Gurney flap configuration shows a notable improvement of 10.28\% in $C_l/C_d$, with a flap height of 1.9\%c and a flap mounting angle of $-58\degree$. The study highlights the effectiveness of the proposed design optimization framework and furnishes valuable insights into optimizing Gurney flap parameters. The comparison of metaheuristic algorithms serves to enhance the study's contribution to Gurney flap design optimization.

"Tyagi, A., Singh, P., Rao, A., Kumar, G. and Singh, R.K., 2023. A Novel Framework for Optimizing Gurney Flaps using RBF Neural Network and Cuckoo Search Algorithm." 
arXiv preprint arXiv:2307.13612.

# Gurney Flap Code Installation

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

Continuous Integration:<br/>
[![Regression Testing](https://github.com/su2code/SU2/workflows/Regression%20Testing/badge.svg?branch=develop)](https://github.com/su2code/SU2/actions)
[![Release](https://github.com/su2code/SU2/workflows/Release%20Management/badge.svg?branch=develop)](https://github.com/su2code/SU2/actions)

Code Quality:<br/>
[![CodeFactor](https://www.codefactor.io/repository/github/su2code/su2/badge)](https://www.codefactor.io/repository/github/su2code/su2)

## Build SU2
The build system of SU2 is based on a combination of [meson](http://mesonbuild.com/) (as the front-end) and [ninja](https://ninja-build.org/) (as the back-end). Meson is an open source build system meant to be both extremely fast, and, even more importantly, as user friendly as possible. Ninja is a small low-level build system with a focus on speed. 

Short summary of the minimal requirements:

- C/C++ compiler
- Python 3

**Note:** all other necessary build tools and dependencies are shipped with the source code or are downloaded automatically.

If you have these tools installed, you can create a configuration using the `meson.py` found in the root source code folder:
```
./meson.py build
```

