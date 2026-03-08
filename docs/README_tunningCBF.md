## Tuning the CBF Gains α₁ and α₂ Using Aircraft Turn Performance

In the second–order CBF condition

$$
\ddot h + 2 \alpha_1 \dot h + \alpha_2 h \ge 0,
$$

the gains $\alpha_1$ and $\alpha_2$ shape how aggressively the system "pushes" the safety margin $h$ away from unsafe values. A convenient way to tune them is to interpret this as a **second–order stable error dynamics**:

$$
s^2 + 2\alpha_1 s + \alpha_2
\quad \Longleftrightarrow \quad
s^2 + 2\zeta \omega_n s + \omega_n^2,
$$

with damping ratio $\zeta$ and natural frequency $\omega_n$. This gives:

$$
\boxed{\alpha_1 = \zeta \,\omega_n, \qquad \alpha_2 = \omega_n^2.}
$$

### Step 1 – Relate ωₙ to Aircraft Turn Limits

For a fixed–wing aircraft, obstacle avoidance is limited by the **minimum turn radius**:

$$
R_{\min} \approx \frac{V^2}{g \tan \phi_{\max}},
$$

where:
- $V$ = airspeed (m/s),
- $g \approx 9.81 \,\text{m/s}^2$,
- $\phi_{\max}$ = maximum allowable bank angle (rad).

A typical **maneuver time scale** for a significant heading change is on the order of the time to traverse one minimum–radius "segment". A simple approximation is:

$$
T_{\text{rec}} \sim \frac{R_{\min}}{V} = \frac{V}{g \tan \phi_{\max}},
$$

where $T_{\text{rec}}$ is the desired "recovery time" over which we want the CBF to significantly bend the trajectory away from danger.

Choose the natural frequency as:

$$
\boxed{\omega_n \approx \frac{2}{T_{\text{rec}}}
= \frac{2 g \tan \phi_{\max}}{V}.}
$$

This ties the CBF aggressiveness directly to what the aircraft can physically do.

### Step 2 – Choose the Damping Ratio ζ

The damping ratio $\zeta$ shapes how "smooth" or "oscillatory" the safety response is:

- $\zeta \approx 0.7$: fast and responsive, slight overshoot.
- $\zeta \approx 1.0$: critically damped, no overshoot, smooth.
- $\zeta > 1.0$: very conservative, slower but safer and less oscillatory.

A good default is:

$$
\boxed{\zeta \in [0.7,\,1.0] \text{ for agile response; } \zeta \in [1.0,\,1.5] \text{ for smoother response}.}
$$

### Step 3 – Compute α₁ and α₂

Given $\omega_n$ and $\zeta$, set

$$
\boxed{
\alpha_1 = \zeta \,\omega_n,
\qquad
\alpha_2 = \omega_n^2.
}
$$

This ensures the CBF inequality corresponds to safety dynamics that are consistent with the aircraft’s turn limits.

### Step 4 – Practical Tuning Procedure

1. **Pick flight condition**: choose representative $V$ and $\phi_{\max}$ for your scenario.
2. **Compute $R_{\min}$**:
   $$
   R_{\min} = \frac{V^2}{g \tan \phi_{\max}}.
   $$
3. **Set recovery time**:
   $$
   T_{\text{rec}} \sim \frac{R_{\min}}{V} = \frac{V}{g \tan \phi_{\max}}.
   $$
4. **Compute $\omega_n$**:
   $$
   \omega_n = \frac{2}{T_{\text{rec}}}.
   $$
5. **Choose $\zeta$** (e.g. $\zeta = 1.0$).
6. **Set $\alpha_1, \alpha_2$** from:
   $$
   \alpha_1 = \zeta \omega_n,\quad \alpha_2 = \omega_n^2.
   $$

### Step 5 – Validation and Refinement

After choosing $\alpha_1, \alpha_2$:

- Simulate worst–case approaches toward obstacles (straight–in, oblique).
- Check:
  - Does the system respect the safety margin (no violations of $h \ge 0$ or chosen buffer)?
  - Is the corrective input $u_{\text{safe}}$ within actuator limits and not overly oscillatory?

If avoidance is too weak or late:
- Increase $\omega_n$ (thus $\alpha_1, \alpha_2$).

If avoidance is too aggressive or chattery:
- Decrease $\omega_n$, or increase $\zeta$.

This loop keeps the CBF behavior physically consistent with the aircraft's turn capability, while allowing you to refine aggressiveness and smoothness through $\omega_n$ and $\zeta$.
