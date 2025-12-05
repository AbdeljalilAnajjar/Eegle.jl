# BCI FII Corpus Benchmark

This document reports the results of a benchmark run on the [FII BCI Corpus](@erf "FII BCI Corpus Overview") using **Eegle**.

* Only subjects/sessions that passed the *first three exclusion criteria* described in the *@discarded.md* file for [MI](https://zenodo.org/records/17801878/files/@discarded.md?download=1) and [P300](https://zenodo.org/records/17793672/files/@discarded.md?download=1) were included in this benchmark.
* Consequently, some sessions exhibit low accuracy scores. These subjects correspond to the *fourth exclusion criterion* mentioned in the *@discarded.md* file. They have been placed in the `/discarded` folder of their respective database and are not good examples to train a machine learning model.
* Individual subject/session accuracies are documented in the `.yml` files and are available in CSV format in this [GitHub repository](https://github.com/FhmDmi/Eegle-Tools/tree/master/Within-Session-Evaluation/results). Scripts used to make this benchmark and a summary of poor performers are also available there.
* This document reports the mean(±sd) accuracies for each database according to the analyzed tasks
    - "right_hand" vs. "feet" and "left_hand" vs. "right_hand" for MI
    - target vs nontarget for P300.

## Available Classifiers

Results are currently available for three Riemannian classifiers:

- **MDM** (Minimum Distance to Mean) with Fisher metric
- **ENLR** (Elastic Net Logistic Regression) with Fisher metric for projecting the data onto the tangent space
- **Linear SVM** (Linear Support Vector Machine) with Fisher metric for projecting the data onto the tangent space (only for MI databases).

---

## Processing Pipelines

For details please see the documentation of the [`crval`](@ref) function.

### Motor Imagery (MI) Pipeline

1. **Bandpass filter:** 8-32 Hz, forward-backward Butterworth filter of order 4.
2. [**Automatic artifact rejection:**](https://marco-congedo.github.io/Eegle.jl/dev/ERPs/#Eegle.ERPs.reject) upperLimit = 1.2
3. **Covariance estimation:** Sample covariance matrix with Tikhonov regularization (λ = 10⁻⁴)
4. **Cross-validation:** Stratified K-Fold (K=10) using seed = 109 for randomization and reproducibility
5. **Evaluation:** Balanced accuracy score for each classifier

### P300 Pipeline

1. **Bandpass filter:** 1-24 Hz, forward-backward Butterworth filter of order 4
2. [**Automatic artifact rejection:**](https://marco-congedo.github.io/Eegle.jl/dev/ERPs/#Eegle.ERPs.reject) upperLimit = 1.2
3. **Covariance estimation:**
   - LWF (Ledoit-Wolf-Framework) shrinkage
   - ERP prototype covariance estimation using super-trial averaging
   - PCA (Principal Component Analysis) dimensionality reduction to 8 components
   - Adaptive weights computed as the inverse of the squared Frobenius norm of the trial data
4. **Cross-validation:** Stratified K-Fold (K=10) using seed = 109 for randomization and reproducibility
5. **Evaluation:** Balanced accuracy score for each classifier

---

## Motor Imagery (MI) Benchmark Results

### Task: right_hand vs. feet

| Database | MDM (%) | ENLR (%) | SVM (%) |
|----------|---------|----------|---------|
| AlexMI-None | 73.75 ± 18.03 | 80.62 ± 14.00 | 75.31 ± 13.39 |
| BNCI2014001-None | 85.13 ± 12.20 | 87.70 ± 9.65 | 86.33 ± 11.05 |
| BNCI2014002-Test | 78.45 ± 17.49 | 79.17 ± 18.49 | 79.64 ± 16.40 |
| BNCI2014002-Train | 77.07 ± 11.40 | 77.54 ± 11.38 | 74.07 ± 13.87 |
| BNCI2015001-None | 81.62 ± 13.08 | 84.53 ± 10.23 | 78.94 ± 12.42 |
| Schirrmeister2017-None | 84.08 ± 14.38 | 94.99 ± 7.08 | 95.88 ± 5.65 |
| Weibo2014-None | 61.19 ± 9.32 | 86.59 ± 8.40 | 81.95 ± 7.88 |
| Zhou2016-None | 87.33 ± 6.76 | 92.89 ± 3.66 | 87.45 ± 8.41 |

### Task: left_hand vs. right_hand

| Database | MDM (%) | ENLR (%) | SVM (%) |
|----------|---------|----------|---------|
| BNCI2014001-None | 77.75 ± 13.99 | 78.26 ± 15.33 | 78.85 ± 14.27 |
| BNCI2014004-Test | 76.67 ± 14.43 | 78.45 ± 14.28 | 76.20 ± 14.93 |
| BNCI2014004-Train | 65.39 ± 10.18 | 67.78 ± 10.27 | 66.31 ± 9.70 |
| Cho2017-None | 60.61 ± 11.17 | 67.43 ± 13.91 | 65.77 ± 11.02 |
| GrosseWentrup2009-None | 59.58 ± 8.08 | 83.11 ± 14.08 | 79.81 ± 12.75 |
| Lee2019_MI-Train | 64.54 ± 13.68 | 76.53 ± 14.74 | 73.25 ± 14.14 |
| PhysionetMI-Task 2 | 47.62 ± 16.07 | 62.95 ± 18.34 | 57.41 ± 12.00 |
| SPSM2025-None | 59.82 ± 9.83 | 63.32 ± 10.41 | 58.38 ± 9.46 |
| Schirrmeister2017-None | 63.85 ± 16.71 | 80.94 ± 14.05 | 79.88 ± 14.32 |
| Shin2017A-None | 57.96 ± 18.48 | 72.14 ± 20.87 | 68.16 ± 18.78 |
| Weibo2014-None | 54.98 ± 11.26 | 79.01 ± 13.61 | 73.51 ± 15.96 |
| Zhou2016-None | 84.52 ± 7.76 | 86.92 ± 8.18 | 82.46 ± 9.31 |

---

## P300 Benchmark Results

| Database | MDM (%) | ENLR (%) |
|----------|---------|----------|
| BNCI2014009-None | 80.11 ± 7.67 | 82.87 ± 6.22 |
| BNCI2015003-Test | 77.18 ± 9.18 | 72.89 ± 6.89 |
| BNCI2015003-Train | 77.76 ± 6.23 | 72.65 ± 6.29 |
| Cattan2019-Personal Computer (PC) | 81.93 ± 7.76 | 82.1 ± 8.36 |
| Cattan2019-Virtual Reality (VR) | 79.07 ± 6.8 | 81.07 ± 6.38 |
| EPFLP300-Run 1 - Television | 68.51 ± 10.75 | 68.75 ± 7.65 |
| EPFLP300-Run 2 - Telephone | 68.25 ± 10.75 | 71.49 ± 8.58 |
| EPFLP300-Run 3 - Lamp | 68.51 ± 13.74 | 71.84 ± 11.07 |
| EPFLP300-Run 4 - Door | 68.06 ± 9.49 | 66.25 ± 7.17 |
| EPFLP300-Run 5 - Window | 69.8 ± 9.17 | 76.18 ± 8.25 |
| EPFLP300-Run 6 - Radio | 67.92 ± 9.77 | 70.4 ± 11.08 |
| bi2012-Online | 77.66 ± 5.15 | 82.9 ± 7.69 |
| bi2012-Training | 73.52 ± 3.28 | 80.55 ± 5.38 |
| bi2013a-Adaptative - Online | 86.95 ± 4.71 | 83.65 ± 6.23 |
| bi2013a-Adaptative - Training | 84.74 ± 5.49 | 84.47 ± 5.51 |
| bi2013a-Non Adaptative - Online | 86.64 ± 6.34 | 83.09 ± 6.27 |
| bi2013a-Non Adaptative - Training | 84.58 ± 5.08 | 84.37 ± 5.28 |
| bi2014a-None | 74.49 ± 7.33 | 77.58 ± 6.66 |
| bi2014b-Solo | 72.76 ± 13.67 | 73.78 ± 12.67 |
| bi2015a-Flash Duration 110ms | 80.01 ± 6.33 | 81.84 ± 6.5 |
| bi2015a-Flash Duration 50ms | 79.96 ± 6.85 | 81.46 ± 6.85 |
| bi2015a-Flash Duration 80ms | 79.38 ± 6.07 | 81.53 ± 6.59 |