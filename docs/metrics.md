# FloodTwin-HIL Metrics Definition

## Depth Metrics
* **RMSE (Root Mean Square Error):** Measures the standard deviation of the prediction errors. Crucial for assessing large errors in depth estimation.
* **MAE (Mean Absolute Error):** Measures the average magnitude of the errors in a set of predictions, without considering their direction.

## Detection Metrics
* **Precision:** The ratio of correctly predicted positive observations to the total predicted positive observations (How many of the detected hazards were actual hazards?).
* **Recall:** The ratio of correctly predicted positive observations to all observations in actual class (How many of the actual hazards did we detect?).
* **F1 Score:** The weighted average of Precision and Recall.

## Reliability Metrics
* **Confidence Stability:** Measures how consistently the system reports its confidence level under varying environmental conditions.

## Robustness Metrics
* **Performance Drop %:** Measures the percentage decrease in performance (e.g., increase in RMSE) when moving from normal conditions to degraded/fault conditions.

## Calibration Metrics
* **Brier Score:** Measures the accuracy of probabilistic predictions. Useful for evaluating the hazard probability estimations.