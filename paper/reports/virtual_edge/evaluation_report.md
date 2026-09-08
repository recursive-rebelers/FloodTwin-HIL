# FloodTwin-HIL Master Evaluation Report

## Analysis Overview

| metric | experiment |
| --- | --- |
| Baseline | baseline_dataset |
| Best F1 | baseline_registry |
| Lowest MAE | baseline_dataset |
| Best ECE | uncertainty_analysis |
| Fastest | baseline_dataset |
| Best overall | baseline_registry |

## Experiment Coverage

| experiment | source_mode | profile_name | input_rows | used_rows | unique_scenarios |
| --- | --- | --- | --- | --- | --- |
| baseline_dataset | dataset | baseline_dataset | 30000 | 30000 | 30000 |
| baseline_registry | registry | baseline_registry | 30000 | 30000 | 30000 |
| complex_environment | dataset | complex_environment | 11179 | 11179 | 11179 |
| compound_fault | dataset | compound_fault | 30000 | 30000 | 30000 |
| flood_severity | dataset | flood_severity | 10049 | 10049 | 10049 |
| high_speed | dataset | high_speed | 10227 | 10227 | 10227 |
| muddy_water | dataset | muddy_water | 11005 | 11005 | 11005 |
| sensor_failure | dataset | sensor_failure | 30000 | 30000 | 30000 |
| stress_test | dataset | stress_test | 30000 | 30000 | 30000 |
| uncertainty_analysis | dataset | uncertainty_analysis | 30000 | 30000 | 30000 |

## Detection Comparison

| experiment | accuracy | precision | recall | specificity | f1 | balanced_accuracy | mcc | roc_auc | pr_auc | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dataset | 0.8829 | 1.0000 | 0.8017 | 1.0000 | 0.8899 | 0.9008 | 0.7896 | 0.9988 | 0.9992 | 0.0303 |
| baseline_registry | 0.8868 | 1.0000 | 0.8082 | 1.0000 | 0.8940 | 0.9041 | 0.7958 | 0.9989 | 0.9993 | 0.0291 |
| complex_environment | 0.7637 | 0.9975 | 0.6045 | 0.9978 | 0.7528 | 0.8011 | 0.6156 | 0.9660 | 0.9773 | 0.0923 |
| compound_fault | 0.7901 | 0.7816 | 0.8944 | 0.6399 | 0.8342 | 0.7671 | 0.5611 | 0.8390 | 0.8571 | 0.1976 |
| flood_severity | 0.8249 | 0.9715 | 0.7391 | 0.9647 | 0.8395 | 0.8519 | 0.6844 | 0.9537 | 0.9711 | 0.0799 |
| high_speed | 0.4093 | — | 0.0000 | 1.0000 | — | 0.5000 | — | 0.8228 | 0.8482 | 0.5904 |
| muddy_water | 0.8200 | 0.9494 | 0.7341 | 0.9437 | 0.8280 | 0.8389 | 0.6692 | 0.9367 | 0.9536 | 0.0996 |
| sensor_failure | 0.7502 | 0.7726 | 0.8175 | 0.6532 | 0.7944 | 0.7354 | 0.4781 | 0.8071 | 0.8317 | 0.2069 |
| stress_test | 0.7764 | 0.7771 | 0.8711 | 0.6400 | 0.8214 | 0.7555 | 0.5313 | 0.8620 | 0.9013 | 0.2013 |
| uncertainty_analysis | 0.8743 | 0.9963 | 0.7901 | 0.9958 | 0.8813 | 0.8929 | 0.7745 | 0.9876 | 0.9917 | 0.0463 |

## Regression Comparison

| experiment | n | mae | rmse | median_ae | max_ae | bias | r2 | mape_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dataset | 30000 | 0.1354 | 0.2111 | 0.0920 | 3.2350 | -0.0237 | 0.9963 | 1.3437 |
| baseline_registry | 30000 | 0.1397 | 0.2157 | 0.0960 | 2.7290 | -0.0241 | 0.9962 | 1.3939 |
| complex_environment | 11179 | 1.0460 | 1.3688 | 0.8320 | 8.4670 | -0.5172 | 0.8444 | 10.1245 |
| compound_fault | 30000 | 1.9388 | 2.6257 | 1.3985 | 12.3450 | 1.0881 | 0.4338 | 22.5629 |
| flood_severity | 10049 | 0.9709 | 1.5123 | 0.5460 | 9.4920 | -0.1149 | 0.7996 | 9.5958 |
| high_speed | 10227 | 5.5491 | 6.4328 | 5.2610 | 14.8790 | -5.5274 | -2.4356 | 45.4982 |
| muddy_water | 11005 | 1.3549 | 1.8488 | 1.0140 | 11.0510 | -0.0841 | 0.7193 | 13.7168 |
| sensor_failure | 30000 | 2.2011 | 2.9354 | 1.6540 | 15.2410 | 0.9846 | 0.2923 | 24.5597 |
| stress_test | 30000 | 2.0659 | 2.5805 | 1.7560 | 10.8900 | 0.8563 | 0.4531 | 23.1286 |
| uncertainty_analysis | 30000 | 0.5585 | 0.7060 | 0.4670 | 4.0770 | -0.0403 | 0.9591 | 5.6852 |

## Uncertainty and Reliability

| experiment | ece | mce | interval_coverage_95 | interval_width_95_mean | winkler_score_95 | confidence_error_pearson | confidence_error_spearman | r_lidar_pearson | r_ultrasonic_pearson | r_radar_pearson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dataset | 0.0767 | 0.3315 | 1.0000 | 4.2562 | 4.2562 | -0.5156 | -0.3796 | -0.3958 | -0.3999 | -0.4404 |
| baseline_registry | 0.0735 | 0.3334 | 1.0000 | 4.1083 | 4.1083 | -0.5172 | -0.3637 | -0.4189 | -0.4166 | -0.4459 |
| complex_environment | 0.0916 | 0.2990 | 0.9391 | 5.0664 | 6.8533 | -0.2376 | -0.1894 | 0.0259 | 0.0089 | -0.0335 |
| compound_fault | 0.1945 | 0.3167 | 0.6762 | 4.7405 | 27.9779 | -0.0330 | 0.0187 | -0.0808 | -0.0569 | -0.0491 |
| flood_severity | 0.0316 | 0.1185 | 0.9120 | 5.1395 | 9.4358 | -0.4407 | -0.4669 | 0.0202 | 0.0290 | 0.0231 |
| high_speed | 0.5905 | 0.5905 | 0.1609 | 4.3897 | 146.2998 | -0.4352 | -0.4542 | -0.0187 | -0.0504 | -0.0871 |
| muddy_water | 0.0416 | 0.1121 | 0.8497 | 4.9066 | 12.5584 | -0.0835 | -0.0812 | 0.0904 | 0.0897 | 0.0819 |
| sensor_failure | 0.1714 | 0.2513 | 0.6199 | 4.8080 | 34.5501 | -0.0219 | -0.0061 | -0.1040 | -0.0953 | -0.0913 |
| stress_test | 0.1831 | 0.3896 | 0.6146 | 4.5840 | 26.0124 | -0.0389 | -0.0334 | 0.0256 | 0.0366 | 0.0383 |
| uncertainty_analysis | 0.0304 | 0.1206 | 0.9979 | 4.3703 | 4.3915 | -0.1406 | -0.0927 | -0.1245 | -0.1248 | -0.1379 |

## Runtime and Ranking

| experiment | avg_total_ms | p95_total_ms | throughput_sps | peak_process_rss_mb | mean_process_cpu_percent | mean_system_cpu_percent | rank_f1 | rank_mae | rank_ece | rank_total_ms | rank_robustness | rank_overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_dataset | 2.6577 | 2.8992 | 376.2649 | 211.4844 | 20.0815 | 1.8349 | 2.0000 | 1.0000 | 5.0000 | 1.0000 | 2.0000 | 2.2000 |
| baseline_registry | 2.8481 | 3.1095 | 351.1165 | 108.3086 | 17.8204 | 1.7243 | 1.0000 | 2.0000 | 4.0000 | 2.0000 | 1.0000 | 2.0000 |
| complex_environment | 3.1880 | 3.4693 | 313.6725 | 129.7383 | 21.3534 | 1.9190 | 9.0000 | 5.0000 | 6.0000 | 9.0000 | 6.0000 | 7.0000 |
| compound_fault | 2.9461 | 3.2130 | 339.4341 | 212.1445 | 21.0816 | 1.8935 | 5.0000 | 7.0000 | 9.0000 | 3.0000 | 7.0000 | 6.2000 |
| flood_severity | 2.9855 | 3.2308 | 334.9541 | 118.6406 | 21.1160 | 1.8807 | 4.0000 | 4.0000 | 2.0000 | 4.0000 | 4.0000 | 3.6000 |
| high_speed | 2.9919 | 3.2648 | 334.2401 | 126.7891 | 21.0118 | 1.8949 | — | 10.0000 | 10.0000 | 5.0000 | 10.0000 | 8.7500 |
| muddy_water | 3.0454 | 3.2925 | 328.3670 | 128.9336 | 21.5368 | 1.9259 | 6.0000 | 6.0000 | 3.0000 | 6.0000 | 5.0000 | 5.2000 |
| sensor_failure | 3.0771 | 3.4210 | 324.9861 | 212.2617 | 21.4008 | 1.9261 | 8.0000 | 9.0000 | 7.0000 | 7.0000 | 9.0000 | 8.0000 |
| stress_test | 3.3485 | 3.6418 | 298.6414 | 200.1406 | 22.1725 | 1.9779 | 7.0000 | 8.0000 | 8.0000 | 10.0000 | 8.0000 | 8.2000 |
| uncertainty_analysis | 3.0914 | 3.3734 | 323.4737 | 200.3594 | 21.6254 | 1.9382 | 3.0000 | 3.0000 | 1.0000 | 8.0000 | 3.0000 | 3.6000 |

