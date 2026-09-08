# Physics-Aware Digital Twin Architecture

### 1. Stage Objective

This stage transforms a synthetic driving scenario into a multi-sensor observation set through independent digital twin models for each sensing modality.

<div align="center">
Scenario Registry<br>
        ↓<br>
Sensor Digital Twins<br>
        ↓<br>
Environmental Degradation & Noise Models<br>
        ↓<br>
Sensor Measurements + Reliability Estimates<br>
        ↓<br>
Synthetic Multi-Sensor Dataset
</div>

Instead of applying a generic noise-injection strategy, the framework models each sensing modality through sensor-specific response functions, environmental degradation mechanisms, and bounded reliability estimation to produce differentiated synthetic measurements.

---

### 2. Sensor Models

#### 2.1 MPU6050 IMU

The IMU model approximates road-induced vehicle motion using surface roughness, environment, water damping, and pothole impact.

**Core Formulation**

- Road roughness is derived from surface and environment factors.
- Water coverage reduces the modeled vibration response through a damping factor.
- Pitch and roll responses are derived from pothole depth within the simplified vehicle-motion model.
- Acceleration is obtained from gravity projection combined with speed-dependent motion, impact response, and noise.
- Reliability decreases as modeled vibration intensity increases.

> **Modeling Note:** The vehicle response is a phenomenological Digital Twin abstraction and does not represent a full vehicle suspension or multibody dynamics model.

#### 2.2 VL53L1X LiDAR

The LiDAR model simulates depth-equivalent optical ranging observations in water-covered potholes.

**Core Formulation**

- Water depth increases modeled optical degradation through an exponential phenomenological relationship.
- Turbidity increases the modeled scattering-related degradation.
- Refraction is represented by a phenomenological depth-dependent bias.
- Gaussian measurement noise increases with modeled environmental degradation.
- Reliability decreases with increasing water depth and turbidity.

> **Modeling Note:** The optical degradation and refraction relationships are engineering-defined phenomenological models rather than experimentally calibrated optical propagation models.

#### 2.3 HLK-LD2411S Radar

The radar model represents a simulated radar-derived depth observation together with environmental sensing-quality indicators.

**Core Formulation**

- Measurement noise increases with water depth and turbidity.
- Clutter probability increases under stronger modeled environmental interference.
- Occasional clutter perturbations are introduced into the simulated distance observation.
- SNR and normalized return-strength indicators are generated as auxiliary quality measures.
- Reliability decreases under increasing water depth and degraded signal quality.

> **Modeling Note:** The HLK-LD2411S is abstracted as a radar sensing modality in the Digital Twin. Its simulated distance output is a depth-equivalent observation rather than a manufacturer-calibrated pothole-depth measurement. The return-strength indicator is not treated as physical radar cross section.

#### 2.4 SEN0189 Turbidity Model

The turbidity module simulates measured NTU values with mild Gaussian disturbance.

**Core Formulation**

- NTU observations are constrained to the configured synthetic operating range.
- Turbidity is classified into discrete water-quality bands.
- Degradation and reliability are represented using exponential phenomenological relationships.

> **Modeling Note:** The turbidity reliability and degradation functions are simulation-defined indicators and are not manufacturer-specified reliability characteristics.

#### 2.5 JSN-SR04T Ultrasonic

The ultrasonic model provides the most structured acoustic observation model in this stage.

**Core Formulation**

- Dry-road behavior produces bottom-echo dominance.
- Wet conditions introduce probabilistic surface-bottom echo competition.
- Bottom-echo response weakens as water depth increases.
- Mixed echoes arise when surface and bottom responses become ambiguous.
- Measurement type is selected from Bottom, Surface, or Mixed.
- Observation quality is represented using echo ratio, effective depth transition, echo clarity, ambiguity, and measurement-error sensitivity.

> **Modeling Note:** Water-layer behavior is represented through phenomenological echo geometry rather than a full acoustic propagation model. The effective blind-zone parameter is a model transition parameter, not a manufacturer-calibrated water-depth limit.

#### 2.6 FSIR01 Water Contact Model

The water-contact sensor provides a simple binary/ordinal water-presence classification layer.

**Core Formulation**

- Effective water depth is perturbed by mild measurement noise.
- Water state is inferred from the resulting effective depth.
- Detection confidence is used as the reliability proxy.

The model provides contextual water-presence information rather than a quantitative pothole-depth measurement.

---

### 3. Reliability Design

Each digital twin produces a sensor observation and, where applicable, a bounded contextual reliability estimate for downstream reliability-aware fusion. Reliability provides the bridge between simulated sensing behavior and adaptive multi-sensor fusion.

**Common Design Principles**

- Higher modeled environmental stress generally reduces sensor reliability.
- Sensor-specific degradation mechanisms are preserved instead of applying one global penalty.
- Confidence and reliability outputs are bounded to maintain numerical stability.
- Measurements and derived quantities are constrained to the configured synthetic operating ranges.
- Sensor-specific stochastic behavior is represented using controlled random variation.

---

### About Dataset (Version 1.0)

The resulting dataset serves as the common observation-level input for the Reliability Engine, Adaptive Fusion Engine, and Fault Injection framework developed in subsequent stages. It contains scenario conditions, synthetic sensor observations, and supporting environmental and sensing-quality variables required for downstream reliability estimation, fusion, and robustness evaluation.