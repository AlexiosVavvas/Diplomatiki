## Comprehensive Document Summary for LLM Reference

### Document Overview
This is a **diploma thesis** from the National Technical University of Athens (NTUA), School of Mechanical Engineering, Control Systems Laboratory. **Title**: "Multi-Robot Collaborative Target Localization using Ergodic Control" by **Alexios Vavvas**, supervised by Prof. Konstantinos J. Kyriakopoulos (Athens, 2025).

**GitHub Repository**: https://github.com/AlexiosVavvas/Diplomatiki

---

### Document Structure & Location Guide

| Chapter | Lines (approx.) | Topic |
|---------|-----------------|-------|
| **Abstract** | 33-78 | English and Greek summaries |
| **Table of Contents** | 90-250 | Full structure listing |
| **Ch 1: Introduction** | 370-400 | Motivation, problem statement, organization |
| **Ch 2: Literature Review** | 401-540 | Multi-agent robotics, ergodic control, safety-critical control |
| **Ch 3: Ergodic Theory** | 540-930 | Core ergodic control methodology (RHEE algorithm) |
| **Ch 4: Target Localization** | 1000-1300 | EKF, bearing-only sensing, Fisher Information |
| **Ch 5: Model Dynamics** | 1310-1700 | Vehicle models (single/double integrator, quadcopter, boat, car, fixed-wing) |
| **Ch 6: Obstacle Avoidance** | 1700-2410 | APF and Control Barrier Functions (CBF) |
| **Ch 7: Implementation** | 2410-2800 | Python/ROS2 system architecture |
| **Ch 8: Results** | 2800-3100 | 7 experimental environments |
| **References** | 3100-3213 | 26 citations |

---

### Key Mathematical Concepts & Their Locations

#### 1. **Ergodic Control Fundamentals** (Lines 544-650)
- **Control-affine dynamics**: $\dot{x}=g(x)+h(x)u$ — Eq. (3.1)
- **Spatial statistics** $C(s,x(t))$: heat-map of agent trajectory — Eq. (3.2)
- **Fourier basis functions** $F_k(s)$: cosine basis for spatial representation — Eq. (3.6)
- **Ergodic metric** $J_\epsilon$: measures trajectory-distribution mismatch — Eq. (3.8)
- **Trajectory coefficients** $c_k$: Fourier coefficients of agent's path — Eq. (3.5)
- **Target distribution coefficients** $\phi_k$: Fourier coefficients of target PDF — Eq. (3.4)

#### 2. **RHEE Algorithm** (Lines 650-780)
- **Mode insertion gradient**: sensitivity of ergodic metric to control — Eq. (3.9)
- **Adjoint equation** for $\rho(t)$: backward-time costate — Eq. (3.10)
- **Optimal control selection**: $u^*(t)=-R^{-1}h(x)^T\rho(t)+u_{def}$ — Eq. (3.12)
- **Application time** $\tau^*$: optimal time to apply control — Eq. (3.13)
- **Control duration** $\lambda$: determined via line search — Eq. (3.15)

#### 3. **Multi-Agent Formulation** (Lines 780-930)
- **Decentralized $c_k$ calculation**: agents share spectral coefficients — Eq. (3.18)
- **Consensus protocol**: additive coefficient sharing — Eq. (3.31)
- **Default vs. Nominal control**: Def. 3.3 and 3.4 (lines 915-925)

#### 4. **Target Localization - EKF** (Lines 1000-1200)
- **Bearing-only measurement model**: azimuth/elevation angles — Eq. (4.4)
- **Measurement Jacobian** $H$: partial derivatives for EKF — Eqs. (4.5-4.11)
- **EKF update equations**: prediction/correction steps — Section 4.4
- **Fisher Information Matrix (FIM)**: quantifies measurement information — Eq. (4.18-4.19)
- **Expected Information Density (EID)**: spatial distribution of info gain — Eq. (4.22)
- **Mahalanobis distance**: data association — Eq. (4.24)
- **Bhattacharyya distance**: target merging criterion — Eq. (4.29)

#### 5. **Vehicle Dynamics Models** (Lines 1310-1700)
- **Single Integrator**: $\dot{x}=u$, 2 states — Section 5.3.1
- **Double Integrator**: position + velocity, 4 states, optional damping — Eq. (5.7)
- **12-DoF Quadcopter**: full rigid-body, Newton-Euler equations — Eqs. (5.8-5.12)
- **Marine Vehicle (Boat)**: 5 states, unicycle-inspired with drag — Eqs. (5.13-5.17)
- **Ground Vehicle (Car)**: 6 states, steering actuator dynamics — Eqs. (5.18-5.23)
- **Fixed-Wing Aircraft**: 12 states, aerodynamic forces/moments — Eqs. (5.24-5.36)

#### 6. **Obstacle Avoidance** (Lines 1700-2410)

**Three approaches**:
1. **Distributional Modification** (Lines 1710-1750): Zero out $\Phi(s)$ in obstacle regions — simple but no collision guarantee
2. **Artificial Potential Fields (APF)** (Lines 1750-1900): Repulsive forces from obstacles — Eqs. (6.3-6.12)
3. **Control Barrier Functions (CBF)** (Lines 1900-2400): Formal safety guarantees

**CBF Key Equations**:
- **Barrier function from APF**: $h(x)=\frac{1}{1+U_{rep}(x)}-\delta$ — Eq. (6.23)
- **CBF condition (relative degree 1)**: $\dot{h}\geq-\alpha(h)$ — Eq. (6.14)
- **Extended CBF (relative degree 2)**: $\ddot{h}\geq-\alpha_1\dot{h}-\alpha_2 h$ — Eq. (6.43)
- **Explicit safe control solution**: Eq. (6.53-6.54)
- **Constraint violation function** $\Psi$: determines if safety intervention needed — Eq. (6.51)
- **Distance functions** for circles, rectangles, walls: Eqs. (6.15-6.20)

---

### Implementation Details (Chapter 7, Lines 2410-2800)

**Core Python Modules**:
- `agent.py`: Central agent controller (ROS2 Node)
- `ergodic_controllers.py`: RHEE algorithm implementation
- `basis.py`: Fourier basis functions with caching
- `model_dynamics.py`: Abstract base + 6 vehicle implementations
- `obstacles.py`: APF + CBF safety systems
- `eid.py`: Multi-target EKF, Fisher Information, sensor models
- `vis.py`: Visualization and plotting

**Key Implementation Features**:
- YAML-based configuration system (agent + obstacle definitions)
- ROS2 distributed architecture with Husarnet VPN networking
- Custom message type `CkCoefficients` for agent communication
- `ReplayBufferFIFO` for trajectory history management
- `ActionMask` for time-based control action sequencing
- Gauss-Legendre quadrature for numerical integration

---

### Experimental Results (Chapter 8, Lines 2800-3100)

| Environment | Description | Key Findings |
|-------------|-------------|--------------|
| **ENV 1** | Single agent, no obstacles | 97% ergodic cost reduction in 20s; demonstrates EKF target localization |
| **ENV 2** | Single/Multi-agent with obstacles | CBF superior to APF (no oscillations); emergent multi-agent coordination |
| **ENV 3** | Tight space (C-shaped) | Finite ergodic memory enables continuous exploration |
| **ENV 4** | Complex maze | 3 agents naturally partition workload without explicit assignment |
| **ENV 5** | Heterogeneous fleet | UGVs, USVs, UAVs coordinate via shared $c_k$ coefficients |
| **ENV 6** | EID updates | Adaptive exploration based on target uncertainty |
| **ENV 7** | Fixed-wing aircraft | 12-DoF nonlinear model; proof-of-concept (not real-time ready) |

---

### Key Parameters & Tuning (Referenced throughout)

| Parameter | Symbol | Purpose | Typical Values |
|-----------|--------|---------|----------------|
| Fourier resolution | $K_{max}$ | Spatial detail level | 4-15 |
| Ergodic memory | $\Delta t_\epsilon$ | History duration | 3-20 seconds |
| Time horizon | $T$ | Prediction window | 0.5-2 seconds |
| Sampling time | $t_s$ | Control update rate | 0.03 seconds |
| CBF damping | $\alpha_1, \alpha_2$ | Safety conservatism | $\alpha_1^2\geq4\alpha_2$ |
| CBF margin | $\delta$ | Safe set size | 0.1 |
| Obstacle influence | $\rho_0$ | APF/CBF range | 0.15-1.5 |

---

### Important Definitions

- **Ergodic trajectory**: $J_\epsilon\to0$ as $t\to\infty$ — trajectory statistics match target distribution
- **Nominal control** $u^{nom}$: Background controller (e.g., LQR for stability)
- **Default control** $u^{def}$: Action mask output combining past optimal actions
- **Safe set** $\mathcal{S}$: $\{x: h(x)\geq0\}$ — forward invariant under CBF
- **Class-$\mathcal{K}$ function**: Continuous, strictly increasing, $\alpha(0)=0$

---

### Limitations & Future Work (Lines 3050-3100)

1. **3D exploration** not fully implemented
2. **Higher relative degree CBF** needed for complex actuators (fixed-wing control surfaces)
3. **Analytical Jacobians** needed for real-time performance on complex models
4. **Adaptive parameter tuning** not yet automated
5. **Obstacle-aware ergodic optimization** would improve trajectory efficiency

---

### How to Find Specific Information

- **Equations**: Search for `\tag{X.Y}` pattern or equation numbers
- **Algorithms**: Algorithm 1 (line ~870), Algorithm 2 (line ~940), Algorithm 3 (line ~960)
- **Figures**: Named as `Figure X.Y` throughout
- **YAML config examples**: Lines 2470-2580 (agent), 2580-2640 (obstacles)
- **Proofs**: Proposition 3.1 (line ~655), Theorem 3.2 (line ~855)
- **Definitions**: Def. 3.3 (line ~910), Def. 3.4 (line ~920)