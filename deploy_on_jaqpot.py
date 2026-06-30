import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

from jaqpotpy import Jaqpot
from jaqpotpy.datasets import JaqpotTabularDataset
from jaqpotpy.models import XGBoostModel
from jaqpotpy.doa import Leverage

# ── 1. Load & preprocess (mirrors the notebook exactly) ──────────────────────

df = pd.read_csv("Cellular_uptake_final.csv")
df = df[df["Incubation Time"] == 24]
df = df.drop(
    columns=[
        "Nanoparticle",
        "Coating",
        "Study",
        "Nominal Size (nm)",
        "Medium type",
        "Medium",
        "Doi",
        "Temperature (oC)",
        "Imputation (Yes/No)",
        "Imputed variables",
        "Cell line",
        "Cell tissue",
        "Method",
        "Incubation Time",
        "Method category",
        "Penicillin/Streptomycin (Yes/No)",
        "Cell species",
        "Cell category",
        "FBS (%)",
        "NP type",
    ]
)

eps = 1e-10
df["log_Cell_uptake"] = np.log(df["Cell uptake"] + eps)
df = df.drop(columns=["Cell uptake"])

# Rename columns: replace spaces and symbols with underscores so Jaqpot accepts them
df.columns = [re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_") for c in df.columns]

# Clean up specific column names
df = df.rename(columns={
    "Sonication_Yes_No": "Sonication",
    "Mean_Hydrodynamic_size_nm": "Mean_Hydrodynamic_size",
    "Ionic_strength_mol_L": "Ionic_strength",
})

# Convert Sonication to categorical
df["Sonication"] = df["Sonication"].map({1: "Sonication", 0: "No_sonication"})

feature_cols = [c for c in df.columns if c != "log_Cell_uptake"]
cat_cols = df[feature_cols].select_dtypes(include="object").columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]

# ── 2. Build JaqpotTabularDataset ─────────────────────────────────────────────

dataset = JaqpotTabularDataset(
    df=df,
    y_cols=["log_Cell_uptake"],
    x_cols=feature_cols,
    task="REGRESSION",
)

# ── 3. Define model & preprocessor ───────────────────────────────────────────

xgb = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=5,
    min_child_weight=1,
    random_state=1234,
)

preprocess_x = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("scaler", StandardScaler(), num_cols),
    ],
)

jaqpot_model = XGBoostModel(
    dataset=dataset,
    model=xgb,
    doa=Leverage(),
    preprocess_x=preprocess_x,
)

# ── 4. Train ──────────────────────────────────────────────────────────────────

jaqpot_model.fit()

# ── 5. Deploy ─────────────────────────────────────────────────────────────────

jaqpot = Jaqpot()
jaqpot.login()

jaqpot.deploy_sklearn_model(
    model=jaqpot_model,
    name="NM Cellular Uptake – XGBoost",
    description=(
        "XGBoost regressor predicting log-transformed cellular uptake of "
        "nanoparticles (24 h incubation). Input features: hydrodynamic size, "
        "zeta potential, concentration, ionic strength, and one-hot encoded "
        "coating type and cell morphology. Trained with StandardScaler "
        "preprocessing and Leverage-based applicability domain."
    ),
    visibility="PRIVATE",
)
