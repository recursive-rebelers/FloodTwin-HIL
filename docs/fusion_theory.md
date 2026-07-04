# Bayesian Fusion Formulation

**Goal:** Infer the true pothole depth ($x$) from uncertain sensor observations ($Z = \{z_{lidar}, z_{ultra}, z_{radar}\}$), while accounting for environmental degradation.

**Methodology:**
We utilize a grid-based discrete Bayesian filter. The state space is defined as depths from 0 to 30 cm.

1. **Prior:** Uniform distribution $P(x)$ assuming no initial bias.
2. **Standard Bayesian Likelihood:** $P(Z|x) \propto P(z_{lidar}|x) \times P(z_{ultra}|x) \times P(z_{radar}|x)$
3. **Adaptive Reliability-Aware Likelihood (Proposed Method):**
   To mitigate degraded sensors poisoning the fusion, we apply Opinion Pooling using the adaptive weights ($w$) derived from our Reliability Engine:
   $P_{adaptive}(Z|x) \propto P(z_{lidar}|x)^{w_{lidar}} \times P(z_{ultra}|x)^{w_{ultra}} \times P(z_{radar}|x)^{w_{radar}}$
   
By exponentiating the likelihood by the reliability weight, a highly unreliable sensor ($w \approx 0$) results in $P(z|x)^0 = 1$, effectively neutralizing its impact on the posterior distribution and preventing failure.