# Adaptive Bayesian Fusion Framework

### 1. Stage Objective

This stage infers pothole depth from heterogeneous sensor observations using reliability-aware generalized Bayesian fusion.

$$
x \in [0,30]\ \text{cm}
$$

$$
Z = \{z_{\text{lidar}}, \, z_{\text{ultra}}, \, z_{\text{radar}}\}
$$

The fusion framework converts contextual sensor reliability, inter-sensor consistency, operational sensor health, and observable environmental context into adaptive sensor trust weights, combining the resulting sensor likelihoods through a reliability-weighted tempered logarithmic opinion pool.

---

### 2. Scene Complexity Input

The adaptive fusion engine accepts a bounded scene-complexity score:

$$
\phi \in [0,1]
$$

where $\phi=0$ represents a relatively simple modeled scene and larger values indicate increased environmental or sensing difficulty.

Scene complexity contributes to the internally computed context-difficulty score and influences the adaptation gate, sensor-specific trust modulation, adaptive uncertainty scaling, and dynamic quality threshold.

---

### 3. Sensor Trust Model

Each sensor reliability $R_i$ is transformed into a nonlinear trust-strength term and subsequently modulated by sensor-specific contextual factors, operational health, and active-sensor consistency.

#### 3.1 Reliability Shaping

The reliability contribution is sharpened using the configured trust exponent:

$$
R_i^{\gamma}
$$

where $R_i$ is the contextual reliability score of sensor $i$ and $\gamma=1.75$.

The sensor prior vector is normalized to unit mean before being applied:

$$
\mathbf{\pi} = \frac{\mathbf{\pi}^*}{\operatorname{mean}(\mathbf{\pi}^*)}.
$$

The current default engineering priors are:

$$
\mathbf{\pi} = [1, 1, 1].
$$

The trust formulation also incorporates bounded effective-sensor and posterior-confidence factors together with sensor health.

#### 3.2 Sensor-Specific Context Modulation

The current implementation uses sensor-specific contextual factors rather than the earlier fixed exponential scene-attenuation equations.

For **LiDAR**:
$$
F_{\text{lidar}} = \operatorname{clip}\left[1 + 0.10A - 0.12\phi - 0.08T - 0.06W - 0.03M - 0.04S - D_L, \, 0.55, \, 1.55\right]
$$

For **Ultrasonic**:
$$
F_{\text{ultra}} = \operatorname{clip}\left[1 + 0.08A - 0.10\phi - 0.10W - 0.05M - 0.04S - 0.50D, \, 0.55, \, 1.55\right]
$$

For **Radar**:
$$
F_{\text{radar}} = \operatorname{clip}\left[1 - 0.01A + 0.09\phi - 0.01T - 0.01W - 0.04M + 0.04S - 0.25D, \, 0.55, \, 1.55\right]
$$

Where:
- $A$ is sensor agreement.
- $\phi$ is scene complexity.
- $T$ is normalized observable turbidity.
- $W$ is normalized observable water-context depth.
- $M$ is normalized measurement spread.
- $S$ is reliability spread.
- $D$ is the normalized dominance penalty derived from the current weight concentration.

For a finite dominance ratio:

$$
D = \operatorname{clip}\left(\frac{D_{\text{ratio}}-1}{3}, \, 0, \, 0.45\right).
$$

The corresponding LiDAR, Ultrasonic, and Radar dominance penalties are scaled by $1.00$, $0.50$, and $0.25$, respectively.

The sensor trust-strength term is then:

$$
T_i = (0.92 + 0.08E_i) \, R_i^{\gamma} \, F_i \, (0.98 + 0.02C_{\text{fusion}})
$$

where $E_i$ is the bounded effective-sensor contribution and $C_{\text{fusion}}$ is the current confidence input to the weighting stage.

Operational sensor health is incorporated as:

$$
T_i \leftarrow T_i \left[h_{\min} + (1 - h_{\min})H_i\right]
$$

where $H_i \in [0,1]$ is sensor health and $h_{\min}=0.08$.

When at least two active sensors are available, the trust term is additionally modulated by the active-valid measurement-consistency score:

$$
T_i \leftarrow T_i \left(C_i\right)^{0.65}.
$$

All trust values are bounded to a numerically valid operating range before normalization.

---

### 4. Adaptive Weight Formation

#### 4.1 Raw Trust Normalization

The trust terms are normalized over the active sensor set to obtain raw sensor weights:

$$
w_i^{\text{raw}} = \frac{T_i}{\sum_{j \in A} T_j}
$$

where $A$ is the set of active sensors. Inactive sensors receive zero weight. If the trust terms are numerically invalid or their sum is negligible, equal weights are assigned over the active sensor set as a safe fallback.

#### 4.2 Observation-Adaptive Uniform Blending

The current implementation blends the raw weights with an active-sensor uniform distribution using an adaptation-dependent retention factor. 

Let the normalized effective-sensor balance over the active sensor count $N_A$ be:

$$
B = \frac{N_{\text{eff}}}{N_A}
$$

The adaptation gate is:

$$
G = \frac{1}{1 + e^{-(d-\tau)/\omega}}
$$

where $d$ is context difficulty, $\tau=0.42$ is the adaptation threshold, and $\omega=0.14$ is the transition width.

The raw-weight retention factor is:

$$
\rho = \operatorname{clip}\left[0.60 + G\left(0.12(0.28)d + 0.05(1-B) - 0.03(A-0.5)\right), \, 0.22, \, 0.78\right].
$$

The active-sensor uniform distribution is $u_i = \frac{1}{N_A}$ and the blended weights are:

$$
w_i \leftarrow \rho w_i^{\text{raw}} + (1-\rho)u_i.
$$

This mechanism limits excessive concentration in a single sensor while preserving observation-dependent reliability structure.

#### 4.3 Minimum Weight Constraint and Quality Gating

A configured weight floor is applied to active sensors during weight processing ($w_{\min}=0.01$). The implementation subsequently evaluates a dynamic sensor-quality score:

$$
Q_i = 0.42R_i + 0.22C_i + 0.16S_i + 0.20H_i
$$

where $C_i$ is measurement consistency, $S_i$ is the sigma-quality score, and $H_i$ is sensor health.

The adaptive quality threshold is:

$$
Q_{\text{th}} = \operatorname{clip}\left[0.24 + 0.12d + 0.08(1-A) + 0.08(1-\bar{C}) + 0.05\sigma_R, \, 0.18, \, 0.45\right]
$$

where $\bar{C}$ is the mean active-sensor consistency and $\sigma_R$ is the reliability-spread input.

Sensors below the dynamic threshold are down-weighted, while sensors within the configured threshold margin are smoothly attenuated. A final participating-sensor mask is then formed. The final participating weights are renormalized before likelihood construction.

---

### 5. Bayesian Fusion

A discrete grid-based generalized Bayesian posterior formulation is used over the modeled depth domain.

#### 5.1 Prior Distribution

The Bayesian fusion core operates over the configured discretized depth domain $x \in [0,30]\ \text{cm}$ with $N=300$ grid samples. A uniform depth prior is used in the posterior construction.

#### 5.2 Sensor Likelihoods and Adaptive Uncertainty

For each participating sensor, a Gaussian depth-observation likelihood is evaluated on the depth grid:

$$
P(z_i \mid x) = \mathcal{N}(x; z_i, \sigma_i^2)
$$

where $z_i$ is the valid sensor observation and $\sigma_i$ is the adaptive likelihood-scale parameter. The adaptive sigma path first establishes a base uncertainty:

$$
\sigma_{i,\text{base}} = \operatorname{clip}\left(\sigma_{i,\text{provided}}, \, 0.45, \, 2.85\right)
$$

When no supplied sigmas are available, the base uncertainty is derived from reliability:

$$
\sigma_{i,\text{base}} = \frac{1}{\sqrt{R_i} + 0.08}.
$$

The context multiplier is $M_{\text{context}} = 1 + G\left(0.45d + 0.05\sigma_R\right)$. Thus:

$$
\sigma_i \leftarrow \sigma_{i,\text{base}} M_{\text{context}}.
$$

The active-valid measurement-consistency penalty is then applied:

$$
\sigma_i \leftarrow \sigma_i \left[1 + 0.22(1 - C_i)\right].
$$

Additional sensor-specific environmental modifiers are applied according to the active adaptation gate ($G$):

**LiDAR:**
$$
\sigma_{\text{lidar}} \leftarrow \sigma_{\text{lidar}} \left[1 + G\left(0.08d + 0.08M + 0.06T + 0.05W + 0.03(1-A)\right)\right]
$$

**Ultrasonic:**
$$
\sigma_{\text{ultra}} \leftarrow \sigma_{\text{ultra}} \left[1 + G\left(0.08d + 0.10M + 0.10W + 0.03(1-A)\right)\right]
$$

**Radar:**
$$
\sigma_{\text{radar}} \leftarrow \sigma_{\text{radar}} \left[1 + G\left(0.07d + 0.12M + 0.06W + 0.04T + 0.03(1-A)\right)\right]
$$

All adaptive sigmas are finally bounded to $0.45 \le \sigma_i \le 2.85$.

#### 5.3 Reliability-Weighted Logarithmic Opinion Pool

The adaptive fusion combines the participating sensor likelihoods using a weighted logarithmic opinion pool:

$$
P_{\text{adaptive}}(Z \mid x) \propto \prod_{i \in A} P(z_i \mid x)^{w_i}
$$

or, equivalently:

$$
\log P_{\text{adaptive}}(Z \mid x) = \sum_{i \in A} w_i \log P(z_i \mid x) + \text{constant}.
$$

This formulation represents generalized/tempered Bayesian pooling rather than a standard conditionally independent product of sensor likelihoods.

#### 5.4 Posterior Estimation

The posterior is obtained from the pooled likelihood and the configured prior:

$$
P(x \mid Z) \propto P(x) \, P_{\text{adaptive}}(Z \mid x)
$$

followed by numerical normalization over the discretized depth grid. The principal depth estimates are:

$$
\hat{x}_{\text{MAP}} = \arg\max_x P(x \mid Z)
$$

$$
\hat{x}_{\mu} = \sum_x x P(x \mid Z).
$$

#### 5.5 Calibration and Evaluation Boundary

The calibration partition is used only to derive fixed base sigmas for the baseline and adaptive fusion configuration. For each sensor:

$$
\sigma_{\text{fixed}} = \operatorname{clip}\left(\sqrt{\operatorname{mean}\left[(z_i - x)^2\right]}, \, \sigma_{\min}, \, \sigma_{\max}\right).
$$

Current fixed-sigma bounds:
- **LiDAR**: $[0.85, 3.50]$
- **Ultrasonic**: $[0.75, 3.20]$
- **Radar**: $[0.60, 2.80]$

Ground-truth depth is accessed in the dedicated calibration operation and later in the evaluation-only scoring stage, but is strictily insulated from the inference path.

---

### 6. Posterior Diagnostics

The fusion framework reports implementation-level diagnostics describing sensor-weight distribution, contextual difficulty, adaptation, posterior concentration, and posterior uncertainty.

#### 6.1 Fusion Weight Entropy
$$
H(\mathbf{w}) = -\sum_i w_i \log(w_i)
$$

#### 6.2 Effective Sensor Count
$$
N_{\text{eff}} = \frac{1}{\sum_i w_i^2}
$$

#### 6.3 Dominance Ratio
$$
D_{\text{ratio}} = \frac{\max(\mathbf{w})}{\operatorname{mean}(\mathbf{w}_{\neg\max})}
$$
*(where $\mathbf{w}_{\neg\max}$ denotes the weights excluding the dominant sensor)*

#### 6.4 Context Difficulty and Adaptation Gate
The engine forms a bounded context-difficulty score:
$$
d = 0.28\phi + 0.18(1-A) + 0.12\sigma_R + 0.10(1-B) + 0.08D_s + 0.08M_s + 0.08W + 0.08T + 0.08(1-C_{\text{fusion}})
$$
The resulting score is clipped to $[0,1]$ and converted into the adaptation gate:
$$
G = \frac{1}{1 + e^{-(d-0.42)/0.14}}.
$$

#### 6.5 Posterior Variance
$$
\sigma_x^2 = \sum_x P(x \mid Z) \left(x - \hat{x}_{\mu}\right)^2 \quad \Rightarrow \quad \sigma_x = \sqrt{\sigma_x^2}.
$$

#### 6.6 Credible Interval
$$
[x_l, x_u]_{0.95} = \operatorname{CI}_{0.95}\left(P(x \mid Z)\right).
$$

#### 6.7 Fusion Confidence
An engineered diagnostic combining entropy concentration, posterior variance, peak, interval width, weight balance, and scene simplicity:

$$
C_{\text{entropy}} = \operatorname{clip}\left(1 - \frac{H(P)}{\log N}, \, 0, \, 1\right)
$$
$$
C_{\text{variance}} = \operatorname{clip}\left(e^{-\sigma_x^2/V_{\text{scale}}}, \, 0, \, 1\right) \quad \text{where} \quad V_{\text{scale}} = 0.12(30-0)^2.
$$
$$
C_{\text{peak}} = \operatorname{clip}\left(\frac{\max(P)}{0.05}, \, 0, \, 1\right).
$$
$$
C_{\text{interval}} = \operatorname{clip}\left(1 - \frac{x_u - x_l}{30-0}, \, 0, \, 1\right).
$$
$$
C_{\text{balance}} = \frac{N_{\text{eff}}}{3}, \quad C_{\text{scene}} = 1 - \phi.
$$

The final fusion confidence diagnostic is:
$$
C_{\text{fusion}} = 0.30C_{\text{entropy}} + 0.18C_{\text{variance}} + 0.20C_{\text{peak}} + 0.12C_{\text{interval}} + 0.20C_{\text{balance}}C_{\text{scene}}.
$$

---

### 7. Adaptive Fusion Processing

The implementation executes the following streamlined inference pipeline:

1. **Initialization & Masking:** Establish active-sensor masks, resolve operational health, and bound reliability values. (Returns NO_VALID_SENSOR if the active mask is empty).
2. **Context & Trust Formulation:** Compute context difficulty and the adaptation gate. Modulate sensor reliability using environmental context, sensor health, and inter-sensor consistency.
3. **Weight Adaptation:** Normalize trust values into raw weights, then blend them toward a uniform distribution using an adaptation-dependent retention factor.
4. **Uncertainty Scaling:** Estimate adaptive sensor uncertainties ($\sigma_i$) by updating calibration baselines with reliability, consistency, and environmental modifiers.
5. **Quality Gating:** Apply dynamic quality thresholding to filter and finalize the participating-sensor set.
6. **Bayesian Pooling:** Combine the participating likelihoods via tempered weighted logarithmic opinion pooling and normalize the resulting posterior.
7. **Estimation & Export:** Compute and export all primary estimates (MAP, expected mean), uncertainty bounds (variance, 95% CI), and system diagnostics.

---

### 8. Adaptive Weight Diagnostics

| Metric | Interpretation |
| :--- | :--- |
| High $N_{\text{eff}}$ | More distributed sensor contribution |
| Low $N_{\text{eff}}$ | More concentrated sensor contribution |
| High $H(\mathbf{w})$ | More evenly distributed weights |
| Low $H(\mathbf{w})$ | More concentrated weights |
| High $D_{\text{ratio}}$ | Stronger dominant-sensor contribution |
| High Adaptation Gate $G$ | Stronger contextual adaptation |
| Low Adaptation Gate $G$ | Greater retention of baseline weighting behavior |
| High Context Difficulty $d$ | More challenging modeled observation context |

These diagnostics characterize the distribution of fusion trust, contextual adaptation, and posterior concentration; they do not by themselves establish estimation accuracy.

---

### 9. Output Role

The fusion framework generates a standardized probabilistic representation used for downstream evaluation, fault-injection robustness testing, virtual Hardware-in-the-Loop (HIL) validation, and benchmarking. 

The evaluation pipeline exports the following metrics:
*   **Depth Estimates:** Adaptive and fixed MAP / posterior-mean depths, alongside absolute estimation errors.
*   **Uncertainty & Scoring:** Posterior variance, entropy, peak probability, 95% credible intervals, NLL, and CRPS.
*   **Fusion Diagnostics:** Adaptive weights, dynamic sigmas, context difficulty, adaptation gate, effective sensor count, and dominance ratio.
*   **Hazard Indicators:** Hazard probability, hazard status, risk score, and hazard-expected depth.

**Methodological Boundary:** Ground-truth depth is strictly reserved for offline calibration and final evaluation scoring. During inference, the Bayesian posterior serves as the sole probabilistic representation of the scene, ensuring rigorous isolation between deployment-time observables and latent evaluation variables.