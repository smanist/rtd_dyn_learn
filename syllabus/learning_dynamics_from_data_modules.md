# Learning Dynamics from Data: Module Outline

## 1. Problem Formulation: What Does It Mean to Learn Dynamics?

**Learning objectives**
- Distinguish among learning a map, vector field, parameter set, closure term, latent state, or control-relevant model.
- Classify dynamics-learning problems by time representation, observation type, model knowledge, inputs, and intended use.
- Recognize the roles of identifiability, observability, excitation, and validation from the start.

**Key concepts / methods**
- Discrete-time maps: $x_{k+1}=F(x_k,u_k;\mu)$.
- Continuous-time vector fields: $\dot{x}=f(x,u;\mu)$.
- Full-state versus partial-state observations: $y_k=h(x_k)+\eta_k$.
- Autonomous, controlled, and parametric systems.
- Known, partially known, and purely unknown models.
- Prediction, explanation, estimation, and control as different learning goals.

---

## 2. Data, Noise, Preprocessing, and Validation Metrics

**Learning objectives**
- Prepare time-series data for dynamics learning.
- Understand how sampling, noise, smoothing, and differentiation affect learned models.
- Select validation metrics appropriate to the intended use of the model.

**Key concepts / methods**
- Sampling rate, aliasing, irregular sampling, missing data.
- Measurement noise versus process noise.
- Smoothing, filtering, denoising, and numerical differentiation.
- One-step prediction error, rollout error, vector-field error.
- Stability, spectral, invariant-measure, and control-performance validation.
- Train/test splits by trajectory, regime, and operating condition.

---

## 3. Linear Full-State Dynamics Identification

**Learning objectives**
- Estimate linear state-space dynamics from full-state data.
- Understand least-squares, regularization, noise bias, conditioning, and stability constraints.
- Interpret learned linear models spectrally and dynamically.

**Key concepts / methods**
- Discrete-time models: $x_{k+1}=Ax_k+Bu_k$.
- Continuous-time models: $\dot{x}=Ax+Bu$.
- Least squares, total least squares, regularized least squares.
- Stable linear model fitting.
- Matrix logarithm viewpoint for continuous-time models.
- DMD only as a special full-state linear identification case.
- Persistence of excitation and rank conditions.

---

## 4. Input-Output and Subspace Identification

**Learning objectives**
- Learn state-space models when the full state is not directly observed.
- Connect input-output data, realization theory, and subspace methods.
- Understand how hidden state dimension, controllability, and observability enter identification.

**Key concepts / methods**
- Linear input-output model: $x_{k+1}=Ax_k+Bu_k$, $y_k=Cx_k+Du_k$.
- Markov parameters and impulse responses.
- Hankel matrices and realization.
- Ho-Kalman, ERA, and N4SID-style subspace identification.
- ARX, ARMAX, and output-error models.
- DMDc as a special full-state controlled identification case.
- Hankel DMD as a special delay/subspace construction.

---

## 5. Continuous-Time Identification and Derivative-Free Formulations

**Learning objectives**
- Identify continuous-time dynamics from sampled data.
- Understand when derivative-based regression is unreliable.
- Use integral, weak-form, and collocation formulations to reduce noise sensitivity.

**Key concepts / methods**
- Vector-field learning: $\dot{x}=f(x,u)$.
- Map learning versus vector-field learning.
- Finite-difference derivative estimation and its limitations.
- Smoothing before differentiation.
- Integral matching: $x(t_{k+1})-x(t_k)=\int_{t_k}^{t_{k+1}} f(x(t),u(t))\,dt$.
- Weak-form identification and test functions.
- Collocation, multiple shooting, and discretization mismatch.

---

## 6. Linear-in-Parameter Nonlinear Dynamics and Sparse Discovery

**Learning objectives**
- Formulate nonlinear dynamics learning as regression over a candidate library.
- Use sparsity and structure to discover interpretable governing equations.
- Understand failure modes due to noise, collinearity, and poor library design.

**Key concepts / methods**
- Library models: $\dot{x}=\Theta(x,u)\xi$ and $x_{k+1}=\Theta(x_k,u_k)\xi$.
- SINDy and SINDy with control.
- Sparse regression, group sparsity, constrained sparse regression.
- Weak-form and integral SINDy.
- Polynomial, trigonometric, rational, and physics-informed libraries.
- Model selection, identifiability, and coefficient uncertainty.

---

## 7. Known Parametric Models and Inverse Problems

**Learning objectives**
- Estimate unknown parameters in known dynamical models.
- Use sensitivity and adjoint methods for gradient-based parameter learning.
- Assess identifiability and uncertainty of learned parameters.

**Key concepts / methods**
- Parametric models: $\dot{x}=f(x,u;\theta)$.
- Nonlinear least squares and prediction-error methods.
- Maximum likelihood estimation.
- Sensitivity equations and adjoint gradients.
- Gauss-Newton and Levenberg-Marquardt methods.
- Fisher information, profile likelihood, and Bayesian parameter inference.
- Structural versus practical identifiability.

---

## 8. Gray-Box Residual and Closure Learning

**Learning objectives**
- Combine known physics with learned correction terms.
- Decide where and how to insert residual models into known dynamics.
- Understand how partial physics can improve sample efficiency and extrapolation.

**Key concepts / methods**
- Residual dynamics: $\dot{x}=f_{\mathrm{known}}(x,u)+g_\theta(x,u)$.
- Unknown constitutive relations, forcing terms, damping terms, closures, and source terms.
- Residual learning, hybrid physics-ML, and gray-box modeling.
- Sparse, neural, and Gaussian-process correction models.
- Conservation-aware and constraint-aware residuals.
- Model-form error and discrepancy modeling.

---

## 9. Incomplete State Models and Non-Markovian Closure

**Learning objectives**
- Recognize when a reduced or partial-state model is not Markovian.
- Formulate latent, memory, or closure models for unresolved state variables.
- Distinguish missing physics from missing state information.

**Key concepts / methods**
- State partitioning: $x=(x_a,x_b)$.
- Known partial dynamics: $\dot{x}_a=f_a(x_a,x_b,u)$ with unknown or unresolved $x_b$.
- Closure models for unresolved variables.
- Latent-state augmentation: $\dot{x}_a=f_a(x_a,z,u)$, $\dot{z}=g_\theta(x_a,z,u)$.
- Memory and non-Markovian effects.
- Delay models, recurrent models, and Mori-Zwanzig-inspired closures.
- When simple residual learning is insufficient.

---

## 10. Black-Box Nonlinear Dynamics Models

**Learning objectives**
- Learn flexible nonlinear dynamics when little model structure is known.
- Compare discrete-time, continuous-time, recurrent, and latent black-box models.
- Understand stability, extrapolation, and rollout-error issues.

**Key concepts / methods**
- Discrete-time neural maps: $x_{k+1}=F_\theta(x_k,u_k,\mu)$.
- Continuous-time neural vector fields: $\dot{x}=f_\theta(x,u,\mu)$.
- Feedforward neural dynamics, residual networks, recurrent neural networks.
- Neural ODEs and latent neural ODEs.
- Gaussian-process and kernel dynamics models.
- One-step loss versus trajectory loss.
- Stability regularization, solver effects, stiffness, and generalization.

---

## 11. Controlled and Parametric Nonlinear Systems

**Learning objectives**
- Learn nonlinear dynamics with inputs and operating parameters.
- Separate autonomous behavior, forced response, and parametric dependence.
- Understand data requirements for controlled and multi-regime systems.

**Key concepts / methods**
- Controlled dynamics: $\dot{x}=f(x,u)$ and $x_{k+1}=F(x_k,u_k)$.
- Control-affine dynamics: $\dot{x}=f(x)+g(x)u$.
- Parametric dynamics: $\dot{x}=f(x,u;\mu)$.
- Nonlinear input-output identification.
- NARX/NARMAX models.
- Bilinear and LPV models.
- Multi-regime and multi-task learning.
- Persistent excitation, closed-loop data bias, and active input design.

---

## 12. Learning Dynamics as State Estimation

**Learning objectives**
- Reinterpret dynamics learning under noisy or partial observations as a state-estimation problem.
- Understand why hidden trajectories become learning variables.
- See how estimation viewpoints improve robustness and uncertainty quantification.

**Key concepts / methods**
- State-space model: $x_{k+1}=F_\theta(x_k,u_k)+w_k$, $y_k=h(x_k)+v_k$.
- Joint optimization over $x_{0:T}$ and $\theta$.
- Filtering versus smoothing.
- Strong-constraint versus weak-constraint formulations.
- Process noise as model error.
- Benefits for partial observations, noisy data, and derivative-free learning.

---

## 13. Classical Linear Gaussian Filtering and Smoothing

**Learning objectives**
- Understand classical filtering and smoothing for linear Gaussian state-space models.
- Compute state estimates, covariance estimates, and observation likelihoods.
- Connect Kalman methods to least squares and system identification.

**Key concepts / methods**
- Linear Gaussian model: $x_{k+1}=Ax_k+Bu_k+w_k$, $y_k=Cx_k+Du_k+v_k$.
- Kalman filter prediction and update steps.
- Kalman gain, innovation, and covariance propagation.
- Rauch-Tung-Striebel smoother.
- Innovation likelihood and maximum likelihood.
- Observability, detectability, and steady-state filtering.

---

## 14. Nonlinear and Ensemble State Estimation

**Learning objectives**
- Extend filtering and smoothing ideas to nonlinear and high-dimensional systems.
- Compare local linearization, sampling, ensemble, and particle-based methods.
- Understand approximation errors in nonlinear state estimation.

**Key concepts / methods**
- Nonlinear model: $x_{k+1}=F(x_k,u_k)+w_k$, $y_k=h(x_k)+v_k$.
- Extended Kalman filter and smoother.
- Unscented Kalman filter.
- Ensemble Kalman filter and ensemble smoother.
- Particle filter.
- Moving-horizon estimation.
- Localization, inflation, degeneracy, and Gaussian approximation limits.

---

## 15. Joint State-Parameter Estimation and EM-Type Learning

**Learning objectives**
- Learn states and parameters simultaneously from noisy or partial observations.
- Interpret system identification as latent-variable inference.
- Use EM, variational, and Laplace viewpoints for dynamics learning.

**Key concepts / methods**
- Unknowns: $x_{0:T}$ and $\theta$.
- Augmented-state filtering and dual estimation.
- Expectation-maximization: smoothing as E-step, parameter update as M-step.
- Iterated EKF/EKS and ensemble smoothers for parameter learning.
- Complete-data likelihood and marginal likelihood.
- MAP estimation, Laplace approximation, and variational inference.
- Identifiability under partial observations.

---

## 16. Learning Dynamics as Trajectory Optimization and Data Assimilation

**Learning objectives**
- Formulate dynamics learning as constrained or penalized trajectory optimization.
- Connect smoothing, data assimilation, and nonlinear least squares.
- Exploit adjoints and sparse structure in long-horizon learning problems.

**Key concepts / methods**
- Strong-constraint objective with dynamics as equality constraints.
- Weak-constraint objective with model-error penalties.
- 3D-Var, 4D-Var, and weak-constraint 4D-Var.
- Multiple shooting and collocation.
- Gauss-Newton smoothing.
- Adjoint gradients.
- Block-tridiagonal structure in trajectory optimization.
- Relationship between EKS and Gauss-Newton methods.

---

## 17. State Reconstruction from Partial Observations

**Learning objectives**
- Build useful state representations from incomplete measurements.
- Understand when delay coordinates or latent states can recover Markovian dynamics.
- Relate partial observation, observability, and representation learning.

**Key concepts / methods**
- Observation model: $y_k=h(x_k)+v_k$.
- Non-Markovian nature of $y_k$ alone.
- Delay coordinates: $z_k=(y_k,y_{k-1},\ldots,y_{k-m})$.
- Takens embedding theorem.
- Subspace identification as state reconstruction.
- Recurrent state estimators.
- Autoencoder and neural state-space latent dynamics.
- Observability versus learnability.

---

## 18. Learning Dynamics as Optimal Control

**Learning objectives**
- Recast dynamics learning as an optimal-control problem over trajectories, parameters, and model-error controls.
- Use adjoint, shooting, and control-variable perspectives to derive learning algorithms.
- Understand how optimal-control theory clarifies long-horizon identification.

**Key concepts / methods**
- Parameter estimation as constrained optimization through dynamics.
- Model-error controls: $x_{k+1}=F_\theta(x_k,u_k)+q_k$.
- Objective terms for observation mismatch and model correction effort.
- Pontryagin and adjoint viewpoints.
- Dynamic programming and HJB connections.
- Differentiable simulation.
- Optimal experiment design and active data acquisition.

---

## 19. Identification for Control and Closed-Loop Learning

**Learning objectives**
- Learn models that are useful for feedback control, not only prediction.
- Understand identification under feedback and safety constraints.
- Evaluate learned models through closed-loop performance.

**Key concepts / methods**
- Identification under closed-loop data.
- Distribution shift induced by feedback.
- Persistent excitation under constraints.
- Adaptive control and dual control.
- Model predictive control with learned models.
- Robust and uncertainty-aware MPC.
- Model-based reinforcement learning.
- Control-oriented validation and safe exploration.

---

## 20. Stochastic Dynamics and Probabilistic Learning

**Learning objectives**
- Learn and reason about stochastic dynamical systems.
- Distinguish process noise, measurement noise, model error, and epistemic uncertainty.
- Produce calibrated probabilistic predictions.

**Key concepts / methods**
- Stochastic difference equations: $x_{k+1}=F_\theta(x_k,u_k)+w_k$.
- Stochastic differential equations: $dx=f_\theta(x,u)\,dt+G_\theta(x,u)\,dW_t$.
- Drift and diffusion estimation.
- Fokker-Planck equation.
- Likelihoods for stochastic dynamics.
- Gaussian-process state-space models.
- Bayesian dynamics models and posterior predictive distributions.
- Uncertainty propagation, calibration, and stochastic stability.

---

## 21. Constraints, Geometry, and Invariance in Learned Dynamics

**Learning objectives**
- Understand how constraints, invariants, symmetries, geometry, and locality change the learning problem.
- Enforce physical or geometric structure by construction, constraint, or regularization.
- Evaluate models based on structure preservation, not just prediction error.

**Key concepts / methods**
- Algebraic constraints: $c(x)=0$ and tangent dynamics $\nabla c(x)^\top f_\theta(x)=0$.
- Invariants: $\nabla C(x)^\top f_\theta(x)=0$.
- Dissipation and Lyapunov structure: $\dot{E}(x)\le 0$.
- Hamiltonian, Lagrangian, port-Hamiltonian, and dissipative systems.
- Symmetry and equivariance: $f(gx)=g f(x)$.
- Dynamics on manifolds: $f_\theta(x)\in T_x\mathcal{M}$.
- Graph locality, permutation equivariance, and PDE locality.
- Structure-preserving validation.

---

## 22. Reduced-Order Dynamics and Closure

**Learning objectives**
- Learn low-dimensional models for high-dimensional dynamical systems.
- Understand projection, latent coordinates, truncation error, and closure.
- Build reduced models suitable for prediction, uncertainty propagation, and control.

**Key concepts / methods**
- Projection-based ROMs: $x\approx V_r a$.
- POD-Galerkin models.
- Balanced truncation.
- Autoencoder reduced coordinates: $z=E(x)$, $x\approx D(z)$.
- Latent dynamics: $\dot{z}=g_\theta(z,u)$.
- Closure error and memory effects.
- Markovian versus non-Markovian ROMs.
- Parametric ROMs.
- Stabilization and control-oriented ROM validation.
