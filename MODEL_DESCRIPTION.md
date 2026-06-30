# Nanoparticle Cellular Uptake Predictor

This model predicts the **cellular uptake of nanoparticles (NPs)** based on their physicochemical properties and experimental conditions. It was developed as part of the diploma thesis of **Katerina Theodoridi** at the National Technical University of Athens (NTUA).

- **Thesis:** Theodoridi, K. (2025). *Machine learning models for predicting the cellular uptake of nanoparticles.* NTUA. [http://dx.doi.org/10.26240/heal.ntua.31351](http://dx.doi.org/10.26240/heal.ntua.31351)
- **GitHub:** [ntua-unit-of-control-and-informatics/nm-cellular-uptake](https://github.com/ntua-unit-of-control-and-informatics/nm-cellular-uptake)

---

## What does this model predict?

The model predicts the **log-transformed cellular uptake** of nanoparticles, expressed in units consistent with the training dataset. The log transformation was applied to the target variable before training to handle the wide range of values in the data.

To interpret the output, apply the exponential function: `Cellular Uptake = exp(model output)`.

---

## Input Features

The model accepts the following input features:

| Feature | Type | Description |
|---|---|---|
| `Mean_Hydrodynamic_size` | Numerical | Mean hydrodynamic diameter of the NP (nm) |
| `Zeta_Potential_mV` | Numerical | Zeta potential of the NP (mV) |
| `Concentration` | Numerical | NP concentration used in the experiment |
| `Ionic_strength` | Numerical | Ionic strength of the medium (mol/L) |
| `Sonication` | Categorical | Whether sonication was applied (`Sonication` / `No_sonication`) |
| `Shape` | Categorical | NP shape (`spherical`, `rod`, `other`) |
| `Coating_type` | Categorical | Type of NP surface coating (`No`, `amine`, `carboxyl`, `biological`, `other`) |
| `Cell_morphology` | Categorical | Morphology of the cell line used (`epithelial`, `endothelial`, `fibroblast`, `macrophage`, `neural`, `other`) |

---

## Model Details

- **Algorithm:** XGBoost Regressor
- **Preprocessing:** One-hot encoding for categorical features; StandardScaler for numerical features
- **Applicability Domain:** Leverage-based method — predictions flagged outside the domain should be interpreted with caution
- **Training data:** Nanoparticle cellular uptake studies filtered to 24-hour incubation time (226 samples)

### Hyperparameters

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `subsample` | 0.7 |
| `colsample_bytree` | 0.7 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 5 |
| `min_child_weight` | 1 |

### Performance (test set, log-transformed scale)

| Metric | Value |
|---|---|
| R² | 0.921 |
| MAE | 1.184 |
| RMSE | 1.768 |

---

## Limitations

- The model was trained exclusively on experiments with a **24-hour incubation time**.
- Predictions for NPs with properties outside the training distribution may be unreliable — always check the applicability domain flag.
- The model does not account for cell line identity, medium composition, or temperature, as these were excluded during feature selection.
