# A Novel Framework for Optimizing Gurney Flaps using RBF Neural Network and Cuckoo Search Algorithm

Enhancing aerodynamic efficiency is vital for optimizing aircraft performance and operational effectiveness. It enables greater speeds and reduced fuel consumption, leading to lower operating costs. Hence, the implementation of Gurney flaps represents a promising avenue for improving airfoil aerodynamics. The optimization of Gurney flaps holds considerable ramifications for improving the lift and stall characteristics of airfoils in aircraft and wind turbine blade designs. The efficacy of implementing Gurney flaps hinges significantly on its design parameters, namely, flap height and mounting angle. This study attempts to optimize these parameters using a design optimization framework, which incorporates training a Radial Basis Function surrogate model based on CFD data from two-dimensional (2D) Reynolds-Averaged Navier-Stokes (RANS) simulations. The Cuckoo Search algorithm is then employed to obtain the optimal design parameters and compared with other competing optimization algorithms. The optimized Gurney flap configuration shows a notable improvement of 10.28\% in $C_l/C_d$, with a flap height of 1.9\%c and a flap mounting angle of $-58\degree$. The study highlights the effectiveness of the proposed design optimization framework and furnishes valuable insights into optimizing Gurney flap parameters. The comparison of metaheuristic algorithms serves to enhance the study's contribution to Gurney flap design optimization.

"Tyagi, A., Singh, P., Rao, A., Kumar, G. and Singh, R.K., 2023. A Novel Framework for Optimizing Gurney Flaps using RBF Neural Network and Cuckoo Search Algorithm." 
arXiv preprint arXiv:2307.13612.

## Gurney Flap Code Installation

Short summary of the minimal requirements:

- C/C++ compiler
- Python 3

**Note:** all other necessary build tools and dependencies are shipped with the source code or are downloaded automatically.
1. In case, you do not have the required dependencies, kindly install using the following code
```
pip install -r requirements.txt
```
2. If you have these dependenies installed, you can create a local repository and use the given code by cloning:
```
git clone https://github.com/GipsyOmega/Surrogate-Metaheuristics.git
```
3. Run the project using
```
python CSASurrogate.py
```
This research paper/project is conducted under the Fluid Mechanics Group of Delhi Technological University, supervised by Prof. XX.
