# PINN

Physics-Informed Neural Networks (PINNs) provide a framework that combines physical
laws (such as the heat equation) with sparse observational data to infer complete spatio
temporal temperature profiles. In this assignment, you will develop a PINN to model heat
distribution in a one-dimensional metal rod using limited, noisy temperature measurements
from a few sensors.

**Problem Description**

Rod length L = 1 m
Thermal diffusivity α = 0.04
Temperature field: u(x,t) governed by:
∂u/∂t = α ∂²u/∂x²

Boundary & initial conditions:
u(0,t) = 0
u(1,t) = 0
u(x,0) = sin(pi * x)

Sensor positions: x = 0.1, 0.3, 0.7 (noisy measurements in heat_sensor_data_sparse.npz)

**PINN Architecture**

Fully connected network: input (x, t), output u_hat(x,t)
Linear(2,64) → Tanh → Linear(64,64) → Tanh → Linear(64,64) → Tanh → Linear(64,1)

**Physics-Informed Loss Function**

Total loss:
L = L_data + L_PDE + L_BC + L_IC
Where:
Data loss: average squared error at sensor points
L_data = (1/Nd) * sum_i (u_hat(xi,ti) - u_obs_i)^2

PDE residual loss: enforces heat equation
L_PDE = (1/Nf) * sum_j (∂u_hat/∂t - α ∂²u_hat/∂x²)_j^2

Boundary condition loss:
L_BC = (1/Nb) * sum_k (u_hat(0,tk)^2 + u_hat(1,tk)^2)

Initial condition loss:
L_IC = (1/Ni) * sum_l (u_hat(xl,0) - sin(pi * xl))^2

Hyperparameters: Nf=5000, Nb=200, Ni=200
Optimizer: Adam, 4000 epochs

**Training & Visualization**

Train on sensor data.
Visualize temperature field with matplotlib colorbar.

Compare to analytical solution:
u(x,t) = exp(-α * pi^2 * t) * sin(pi * x)

Plot at times: t = 0, 0.25, 0.5, 0.75, 1.0

Repeat training with only L_data (no physics), observe differences.

**Part B: Making PINNs Explainable**

Generate dense temperature samples from trained PINN.
Fit interpretable models:
Linear Regression
Decision Tree
