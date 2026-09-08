# Fault Injection and Robustness Validation

### 1. Stage Objective

The implementation couples synthetic fault injection with reliability degradation, uncertainty inflation, context recomputation, and hazard-aware robustness validation.

$$
\text{Dataset} \rightarrow \text{Fault Injection} \rightarrow \text{Context Recompute} \rightarrow \text{Adaptive Fusion} \rightarrow \text{Robustness Metrics}
$$

Faulted measurements, reliabilities, availability masks, and uncertainty estimates are propagated into both fixed and adaptive fusion for controlled robustness comparison.

---

### 2. Fault Taxonomy

#### 2.1 Dropout

A Bernoulli missing-measurement process is applied to the selected sensor stream.

$$
M_t \sim \mathrm{Bernoulli}(p_d)
$$

$$
\hat{z}_t =
\begin{cases}
\mathrm{NaN}, & M_t = 1 \\
z_t, & M_t = 0
\end{cases}
$$

Dropout severity determines the missing probability; evaluated cases include 10% and 50%.

#### 2.2 Additive Noise

Synthetic Gaussian corruption is injected into the measurement stream.

$$
\hat{z}_t = z_t + \epsilon_t
$$

$$
\epsilon_t \sim \mathcal{N}(0,\sigma_n^2)
$$

The injected measurement is clipped to the valid non-negative measurement domain.

#### 2.3 Communication Delay

A synthetic causal frame displacement represents communication latency.

$$
\hat{z}_t = z_{t-\Delta}
$$

$$
\Delta = \left\lfloor \frac{d_{ms}}{1000/f_s} \right\rceil
$$

The current implementation evaluates a high-delay scenario using a 20 FPS stream model.

#### 2.4 Synchronization Error

A synthetic temporal frame offset represents inter-sensor synchronization error.

$$
\hat{z}_t = z_{t-\Omega}
$$

The evaluated synchronization case applies a high backward frame offset.

#### 2.5 Drift

A temporal drift model is reserved for future evaluation.

$$
\hat{z}_t = z_t + b_0 + rt
$$

$$
\hat{z}_t = z_t + b_0 + rt + \sum_{k=1}^{t} \eta_k
$$

Drift is not part of the current reported robustness benchmark.

---

### 3. Fault Effect Propagation Model

Non-dropout faults propagate through the FaultEffects layer as multiplicative reliability and uncertainty modifiers.

$$
R_i' = \mathrm{clip}(R_i \rho_i, 0, 1)
$$

$$
\sigma_i' = \sigma_i \lambda_i
$$

where $\rho_i$ and $\lambda_i$ are fault-dependent reliability and uncertainty factors returned by the fault-effect model.

For dropout, the affected samples become unavailable and their reliability is forced to zero.

$$
R_i' =
\begin{cases}
0, & \text{dropped or invalid} \\
R_i, & \text{otherwise}
\end{cases}
$$

All fault-modified measurements are revalidated through finite-value masking before fusion.

---

### 4. Sensor-Specific Fault Response

#### 4.1 LiDAR

LiDAR is evaluated primarily under dropout and compound degradation.

$$
\hat{z}^{\text{LiDAR}}_t \Rightarrow R'_{\text{LiDAR}}, \sigma'_{\text{LiDAR}}
$$

#### 4.2 Ultrasonic

Ultrasonic is evaluated under dropout, delay, and compound degradation.

$$
\hat{z}^{\text{Ultra}}_t \Rightarrow R'_{\text{Ultra}}, \sigma'_{\text{Ultra}}
$$

#### 4.3 Radar

Radar is evaluated under dropout, severe noise, synchronization error, and compound degradation.

$$
\hat{z}^{\text{Radar}}_t \Rightarrow R'_{\text{Radar}}, \sigma'_{\text{Radar}}
$$

---

### 5. Context Re-Estimation After Fault Injection

The fault-corrupted measurements and reliabilities are used to recompute the adaptive fusion context.

###### Measurement Spread

For two valid sensors:

$$
S_m = 0.5|z_1-z_2|
$$

For three valid sensors:

$$
S_m = 1.4826\cdot\mathrm{MAD}(z)
$$

With fewer than two valid sensors, $S_m=0$.

###### Sensor Agreement

For two or more valid sensors:

$$
A_s = \exp\left[-0.25\left(\frac{\bar{d}}{\max(1.4826\mathrm{MAD}(z),0.35)}\right)^2\right]
$$

where $\bar{d}$ is the mean pairwise absolute measurement difference.

###### Clutter Proxy

$$
C = 0.30W + 0.35N + 0.20S_n + 0.15WN
$$

where $W$ is normalized water depth, $N$ is normalized NTU, and $S_n$ is the normalized measurement spread.

###### Scene Complexity

$$
\phi = 0.26W_s + 0.24N_s + 0.18C_s + 0.20S_s + 0.12V_s
$$

where the terms are the square-root normalized water depth, NTU, clutter, measurement spread, and severity level.

###### Reliability Spread

$$
\Delta_R = R_{\max}-R_{\min}
$$

computed over active sensors.

###### Effective Sensor Count

$$
N_{\mathrm{eff}} = \frac{1}{\sum_i w_i^2}
$$

The context is recomputed separately for every fault experiment rather than reused from the clean stream.

---

### 6. Adaptive Fusion Under Faults

Fault-corrupted measurements are fused with fault-adjusted reliabilities, adaptive consistency, sensor-quality gating, and adaptive uncertainty.

$$
P(Z \mid x) \propto \prod_{i \in \{l,u,r\}} P(z_i \mid x)^{w_i}
$$

Raw trust follows the generalized form:

$$
T_i \propto \pi_i R_i^{\gamma} H_i\,C_i^{0.65}
$$

followed by active-sensor normalization and bounded uniform blending:

$$
w_i = \rho\,w_i^{\mathrm{raw}} + (1-\rho)\frac{1}{N_{\mathrm{active}}}
$$

where

$$
\rho = \mathrm{clip}\left(
\mathrm{weight\_blend}
+G_{\phi}\left[
0.12\,\mathrm{context\_blend\_boost}\,\phi
+0.05(1-B)-0.03(A_s-0.5)
\right],\,0.22,\,0.78\right)
$$

and $B = N_{\mathrm{eff}}/3$.

Sensor quality is computed as:

$$
Q_i = 0.42R_i + 0.22C_i + 0.16Q_{\sigma,i} + 0.20H_i
$$

with

$$
Q_{\sigma,i} = 1 - \mathrm{clip}\left(
\frac{\sigma_i-\sigma_{\min}}{\sigma_{\max}-\sigma_{\min}},0,1
\right)
$$

The adaptive quality threshold is:

$$
\tau_q = \mathrm{clip}\left(
q_0+0.12\phi+0.08(1-A_s)+0.08(1-\bar{C})+0.05\Delta_R,
0.18,0.45\right)
$$

The final posterior is obtained from the discrete Bayesian depth grid:

$$
P(x \mid Z) \propto P(Z \mid x)P(x)
$$

No-valid-sensor samples return a fusion failure rather than fabricating a measurement.

---

### 7. Robustness Metrics

###### Estimation Error

$$
\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (\hat{x}_t - x_t)^2}
$$

$$
\mathrm{MAE} = \frac{1}{N} \sum_{t=1}^{N} |\hat{x}_t - x_t|
$$

###### Relative Gain

$$
\mathrm{Gain}_{\%} =
\frac{\mathrm{RMSE}_{\mathrm{fixed}}-\mathrm{RMSE}_{\mathrm{adaptive}}}
{\mathrm{RMSE}_{\mathrm{fixed}}} \times 100
$$

###### Win Rate

$$
\mathrm{Win\ Rate}_{\%} =
\frac{1}{N} \sum_{t=1}^{N}
\mathbb{I}\left(e_t^{\mathrm{adaptive}} < e_t^{\mathrm{fixed}}\right) \times 100
$$

###### Hazard Metrics

Hazard probability is produced by the HazardAssessment module from the fused posterior, fusion confidence, and scene complexity.

$$
P_{\mathrm{hazard}} =
f\left(P(x\mid Z),C_{\mathrm{fusion}},\phi\right)
$$

The reported rate is the proportion of predictions classified as **HAZARD**.

$$
\mathrm{Predicted\ Hazard\ Rate}_{\%} =
\frac{\#_{\mathrm{predicted\ HAZARD}}}{N} \times 100
$$

This quantity is a predicted hazard prevalence metric, not a classification hit rate.

---

### 8. Fault Scenarios

- Baseline (No Faults)
- LiDAR Low Dropout
- LiDAR High Dropout
- Ultrasonic High Dropout
- Radar High Dropout
- Radar Severe Noise
- Ultrasonic High Delay
- Radar High Sync Error
- Compound Scenario A
- Compound Scenario C

---

### 9. Calibration Strategy

1. A 20% calibration subset is separated from an 80% evaluation subset using random seed 42 and disjoint scenario IDs.
2. Fixed sigma values are derived only from the calibration subset using robust residual dispersion:
   $$
   \sigma_{\mathrm{fixed}} = \mathrm{clip}\left(1.4826\cdot\mathrm{median}|z-x|+0.20, \sigma_{\min}, \sigma_{\max}\right)
   $$
3. The calibrated values are used as fixed-fusion base sigmas; they are not treated as manufacturer sensor specifications.
4. Fault effects modify measurements, reliabilities, availability masks, and fault-adjusted sigmas before adaptive fusion.
5. Faulted context is recomputed from the corrupted streams and active reliabilities for every experiment.
6. Robustness is reported through RMSE, MAE, adaptive gain, win rate, failure rate, confidence, entropy, variance, credible-interval width, and hazard response.
7. Truth variables are used only for calibration/evaluation metrics; they are not supplied to inference-time fusion.

---

### 10. Output Role

The robustness pipeline produces a controlled fault-aware benchmark for comparing fixed and adaptive fusion under physically motivated synthetic sensor and communication degradation. The diagnostics provide the robustness-validation layer for subsequent evaluation and HIL-oriented development.