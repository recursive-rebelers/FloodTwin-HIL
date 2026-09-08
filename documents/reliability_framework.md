# Sensor Reliability Framework

### 1. Reliability Definition

For each sensor $s \in \{\text{LiDAR}, \text{Ultrasonic}, \text{Radar}\}$, reliability is modeled as a bounded score:

$$
R_s \in [0,1]
$$

where $R_s$ denotes a normalized contextual trust score representing the expected consistency of a sensor observation with the modeled scene under the prevailing environmental conditions. The reliability score is an inference-time contextual diagnostic and is not interpreted as a calibrated probability of measurement correctness.

---

### 2. LiDAR Reliability Model

LiDAR reliability is driven by optical transmission loss, turbidity, observable water-context severity, and effective measurement noise.

$$
L_{\text{opt}} = \alpha_t \frac{\text{NTU}}{\text{NTU}_{\max}} + \alpha_d d_w
$$

$$
T = e^{-L_{\text{opt}}}
$$

$$
\sigma_{\text{eff}} = \sigma_n \left(1 + 0.45(1 - T)\right)
$$

$$
P_n = \exp\left[-0.18\left(\frac{\sigma_{\text{eff}}}{\sigma_{\text{ref}}}\right)^{1.15}\right]
$$

$$
P_{\text{int}} = \exp\left[-1.15 \cdot \gamma \cdot \frac{\text{NTU}}{\text{NTU}_{\max}} \cdot \frac{d_w}{20}\right]
$$

$$
R_{\text{LiDAR}} = \mathrm{clip}\left(T \cdot P_n \cdot P_{\text{int}}, 0.10, 1.0\right)
$$

Here, $d_w$ denotes an observable water-context depth proxy supplied to the reliability model rather than latent ground-truth water depth.

#### Reliability Bands

| Range | Label |
| :--- | :--- |
| $R \ge 0.80$ | High |
| $0.60 \le R < 0.80$ | Moderate |
| $0.35 \le R < 0.60$ | Low |
| $R < 0.35$ | Very Low |

---

### 3. Ultrasonic Reliability Model

Ultrasonic reliability is modeled using composite bottom-echo quality, effective blind-zone response, echo ambiguity, and phenomenological water-depth attenuation.

$$
E = E_s + E_b + \varepsilon
$$

$$
\rho = \frac{E_b}{E_s + E_b + \varepsilon}
$$

$$
Q_E = 0.55\sqrt{\frac{E_b}{E_s + E_b + \varepsilon}} + 0.45\sqrt{E_b}
$$

$$
D(d_w) = 0.88 + 0.12\tanh\left(\frac{d_w}{d_{bz}}\right)
$$

$$
A = 1 - \frac{|E_b - E_s|}{E_s + E_b + \varepsilon}
$$

$$
P_A = e^{-\beta A}, \qquad P_W = e^{-\lambda d_w}
$$

$$
R_{\text{Ultrasonic}} = \mathrm{clip}\left(Q_E \cdot D \cdot P_A \cdot P_W, 0.25, 0.88\right)
$$

The composite bottom-echo quality term $Q_E$ incorporates both bottom-echo dominance and absolute bottom-echo strength. The effective depth-response term is bounded and does not collapse the reliability score in the absence of water.

#### Reliability Bands

| Range | Label |
| :--- | :--- |
| $R \ge 0.80$ | High |
| $0.60 \le R < 0.80$ | Moderate |
| $0.40 \le R < 0.60$ | Low |
| $R < 0.40$ | Very Low |

---

### 4. Radar Reliability Model

Radar reliability is modeled with slower environmental degradation than the optical channel, reflecting the intended robustness of the simulated radar modality under shallow-water conditions.

$$
F_{\text{SNR}} = \sigma\left(\frac{\text{SNR} - 0.48 \cdot \text{SNR}_{\text{nom}}}{0.22 \cdot \text{SNR}_{\text{nom}}}\right)
$$

$$
F_C = \sigma\left(\frac{0.62 \cdot C_{\max} - C}{0.16 \cdot C_{\max}}\right)
$$

$$
\lambda_W = 0.013 + 0.004(1 - F_{\text{SNR}}) + 0.003\frac{C}{C_{\max}}
$$

$$
P_W = e^{-\lambda_W d_w}
$$

$$
Q = 0.0038 \cdot C^{1.25} \frac{d_w}{C_{\max}}
$$

$$
I = 1 + \eta(1 - F_{\text{SNR}})
$$

$$
P_I = e^{-QI}
$$

$$
F_{\text{return}} = 0.85 + 0.15\frac{r}{r + 0.35}
$$

$$
R_{\text{Radar}} = \mathrm{clip}\left(F_{\text{SNR}} \cdot F_C \cdot P_W \cdot P_I \cdot F_{\text{return}}, 0.12, 1.0\right)
$$

Here, $d_w$ represents the observable water-context proxy and $r$ denotes the normalized return-strength/RCS-related observable used by the phenomenological model. The radar formulation is intentionally less sensitive to water-depth degradation than the optical channel.

#### Reliability Bands

| Range | Label |
| :--- | :--- |
| $R \ge 0.80$ | High |
| $0.60 \le R < 0.80$ | Moderate |
| $0.40 \le R < 0.60$ | Low |
| $R < 0.40$ | Very Low |

---

### 5. Reliability Fusion

The adaptive reliability fusion framework converts sensor-specific contextual reliability scores into normalized trust weights while incorporating scene complexity and inter-sensor reliability disagreement.

#### 5.1 Trust Formation

$$
T_s = \pi_s \cdot \tilde{R}_s^{\gamma}
$$

where $\tilde{R}_s$ is the sanitized reliability score and $\pi_s$ is the sensor prior normalized to unit mean.

#### 5.2 Weight Normalization

$$
w_s = \frac{T_s}{\sum_j T_j}
$$

Only sensors with valid observations contribute to the normalization. If no valid sensor is available, all sensor weights are set to zero and fusion is explicitly treated as unavailable rather than as a physical zero measurement.

#### 5.3 Scene-aware Attenuation

$$
T_{\text{LiDAR}} \leftarrow T_{\text{LiDAR}} e^{-0.12 \phi}
$$

$$
T_{\text{Ultrasonic}} \leftarrow T_{\text{Ultrasonic}} e^{-0.11 \phi}
$$

$$
T_{\text{Radar}} \leftarrow T_{\text{Radar}} e^{-0.05 \phi}
$$

where $\phi$ is the scene complexity score derived from observable contextual quantities, including water-context severity, turbidity, radar clutter, inter-sensor measurement disagreement, SNR difficulty, and echo ambiguity.

#### 5.4 Consensus Regularization

Let $\bar{R}$ denote the mean reliability and $\sigma_R$ the reliability spread. The consensus regularization factor is computed as:

$$
\delta = \frac{\sigma_R}{\bar{R} + \varepsilon}
$$

$$
\alpha = \mathrm{clip}\left(\beta_0 + \beta_1\left(1 - e^{-\delta(1 + 0.5\phi)}\right), 0, 0.28\right)
$$

$$
\mathbf{w} \leftarrow (1-\alpha)\mathbf{w} + \alpha\mathbf{u}
$$

where $\mathbf{u} = [\frac{1}{3}, \frac{1}{3}, \frac{1}{3}]$ for the three-sensor case.

Consensus regularization is an engineering-defined mechanism that limits excessive dominance by a single sensor when inter-sensor reliability disagreement becomes large. The regularization is applied to active sensors and does not assign non-zero trust to an unavailable measurement.

---

### 6. Diagnostic Metrics

The fusion engine reports the following implementation-level diagnostics:

$$
H(\mathbf{w}) = -\sum_i w_i \log(w_i)
$$

$$
N_{\mathrm{eff}} = \frac{1}{\sum_i w_i^2}
$$

$$
D_{\mathrm{ratio}} = \frac{\max(\mathbf{w})}{\mathrm{mean}(\mathbf{w}_{\neg \max})}
$$

For a single active sensor, $D_{\mathrm{ratio}}$ is reported as $\infty$; for no active sensors it is reported as $0$.

$$
C_{\mathrm{balance}} = \frac{N_{\mathrm{eff}}}{N}
$$

where $N$ is the number of sensors considered in the active fusion configuration.

$$
C_{\mathrm{fusion}} = 0.40 \cdot C_{\mathrm{balance}} + 0.40 \cdot \bar{R} + 0.20 \cdot (1-\phi)
$$

$C_{\mathrm{fusion}}$ is a normalized fusion confidence score for system diagnostics and is not a calibrated probability of estimation correctness.

Additional diagnostics include reliability spread, reliability entropy, measurement spread, measurement consistency, effective sensor count, dominant sensor, and adaptive-versus-equal fusion error metrics.

---

### 7. Calibration Strategy

The implementation is calibrated by:

1. **Prior Normalization**: Normalize sensor priors to unit mean.
2. **Boundary Enforcement**: Bound reliability values to the configured operating range.
3. **Contextual Degradation**: Apply sensor-specific phenomenological degradation penalties using observable contextual variables.
4. **Availability Tracking**: Explicitly track sensor availability and prevent unavailable measurements from contributing to reliability weights or fusion.
5. **Complexity Adjustment**: Apply scene-aware trust attenuation.
6. **Consensus Adjustment**: Apply consensus regularization according to inter-sensor reliability disagreement.
7. **Diagnostic Computation**: Compute entropy, effective sensor count, dominance ratio, and fusion confidence.
8. **Baseline Evaluation**: Evaluate adaptive fusion against equal-weight and oracle best-sensor reference baselines.

The framework maintains a strict separation between latent truth variables used for evaluation and observable/contextual variables used during inference. Ground-truth depth, ground-truth water depth, ground-truth turbidity, and severity labels are not supplied to the reliability, context, weighting, or fusion computations.

---

### About Dataset (Version 2.0)

The resulting reliability-enhanced Dataset Version 2.0 serves as the standardized intermediate representation for Adaptive Fusion, Fault Injection, Hardware-in-the-Loop validation, and all subsequent evaluation stages of the pipeline. The dataset preserves the separation between latent ground truth and inference-time observables.