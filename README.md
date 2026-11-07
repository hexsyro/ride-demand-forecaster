# 🚲 Ride Demand Forecaster

**Can careful feature engineering alone turn a weak model into a strong one?**

This project uses a bike-share demand dataset (timestamp, weather conditions, temperature, humidity, wind speed, holiday and weekend flags, season, ride count) to predict hourly ride volume. The point isn't the model, it's the feature engineering. The same Random Forest regressor (250 trees, fixed seed) is trained on four versions of the data (raw, encoded, scaled, and fully engineered) so the RMSE change at each stage is visible and reproducible.

---

![Test RMSE of the same Random Forest model at each data preparation stage](outputs/figures/rmse_by_stage.png)

## 🎯 Project Questions

- How much does a raw, minimally prepared table limit a model, compared with a properly cleaned and encoded one?
- Does one-hot encoding the coded categorical columns (`weather_code`, `season`) help, compared with leaving them as plain numbers?
- Does scaling change results for a model that doesn't strictly require it (and would it matter more for one that does)?
- How much predictive power is hidden inside a raw timestamp once it's decomposed into hour, day of week, month, weekend, rush hour and days since start?
- Do interaction and polynomial terms (for example temperature × humidity) earn their place in a tree-based model?

## Findings

1. **Hour of day is the biggest lever:** test RMSE fell from 952.5 on the raw table to 250.2 on the encoded table, a drop of about 74%. The raw table has no `hour` column, while the encoded table adds it (and one-hot encodes `weather_code` and `season`), so this stage change combines both steps.
2. **Scaling changes nothing for a forest:** Encoded (250.2) and Scaled (250.0) are a tie, well within run-to-run noise. Tree models only compare values within a column, so rescaling a column doesn't change their splits.
3. **Datetime features give a real gain:** adding day of week, month, weekend, rush-hour and days-since-start features, plus interaction/polynomial terms, brought RMSE from 250.0 to 221.5 (about 11% lower).
4. **Demand has two very different daily rhythms:** on weekdays, average rides peak sharply around 8am (roughly 3,800) and again around 5 to 6pm (roughly 3,200), which matches commuting. On weekends there is one smooth hump from late morning to mid-afternoon (roughly 2,100 at its peak).
5. **Polynomial terms alone did not help the forest:** in the notebook experiments, adding squared and product terms on their own made the error slightly worse, which is expected because trees find curves and interactions by splitting. See `notebooks/06_datetime_features.ipynb` for the open question.
6. **Treat the best score as optimistic:** the split is random, so a test hour can have its neighboring hours in the training set, and `days_since_start` lets the forest exploit that. A true forecast should train on earlier data and test on later data (see Possible Extensions).

---

## 📁 Repository Structure

```
ride-demand-forecaster/
├── data/
│   ├── raw/                  # Untouched source demand dataset (store_sharing.csv)
│   └── processed/            # Encoded, scaled and fully feature-engineered data
│
├── notebooks/                 # Step-by-step, one notebook per phase
│   ├── 01_load_explore.ipynb
│   ├── 02_missing_data_imputation.ipynb
│   ├── 03_encoding_categoricals.ipynb
│   ├── 04_feature_scaling.ipynb
│   ├── 05_interaction_polynomial_features.ipynb
│   ├── 06_datetime_features.ipynb
│   └── 07_model_comparison_before_after.ipynb
│
├── src/                        # Reusable code, imported by notebooks/scripts (not copy-pasted)
│   ├── __init__.py
│   ├── load_encode.py                 # Loads raw data, maps codes to names, one-hot encodes categoricals
│   ├── scale.py                   # StandardScaler helpers for the numeric columns
│   ├── features.py                 # Interaction, polynomial, and datetime feature extraction, plus plots
│   └── train_test.py                     # Trains the same model on each feature-engineering stage
│
├── tests/                      # Unit tests for src/ functions
│   ├── __init__.py
│   └── test_features.py
│
├── models/                     # Saved trained models per stage (.joblib), gitignored
│
├── outputs/
│   └── figures/                  # RMSE-by-stage chart, hourly demand and trend plots
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup

```bash
git clone https://github.com/hexsyro/ride-demand-forecaster
cd ride-demand-forecaster
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

1. **Add the raw dataset.** Download the [Bike Demand Dataset](https://www.kaggle.com/datasets/huseyincenik/bike-demand-capstone-project/data) from Kaggle and save it as `data/raw/store_sharing.csv`. Then run the pipeline in order:
   ```bash
   python -m src.load_encode
   python -m src.scale
   python -m src.features
   python -m src.train_test
   ```
2. **Explore in notebooks**, in order, starting with `notebooks/01_load_explore.ipynb`
   | Notebook | Open |
   |---|---|
   | 01 Load and explore | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hexsyro/ride-demand-forecaster/blob/main/notebooks/01_load_explore.ipynb) |
   | 02 Missing data imputation | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hexsyro/ride-demand-forecaster/blob/main/notebooks/02_missing_data_imputation.ipynb) |
   | 03 Encoding categoricals | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hexsyro/ride-demand-forecaster/blob/main/notebooks/03_encoding_categoricals.ipynb) |
   | 04 Feature scaling | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hexsyro/ride-demand-forecaster/blob/main/notebooks/04_feature_scaling.ipynb) |
   | 05 Interaction and polynomial features | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hexsyro/ride-demand-forecaster/blob/main/notebooks/05_interaction_polynomial_features.ipynb) |
   | 06 Datetime features | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hexsyro/ride-demand-forecaster/blob/main/notebooks/06_datetime_features.ipynb) |
   | 07 Model comparison before and after | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/hexsyro/ride-demand-forecaster/blob/main/notebooks/07_model_comparison_before_after.ipynb) |
3. Each notebook (02-06) adds one feature-engineering technique and saves an intermediate
   dataset to `data/processed/`
4. `notebooks/07_model_comparison_before_after.ipynb` trains the same model on each stage
   and plots RMSE improvement

---

## 📊 Skills Practiced

| Skill                                            | Where                                  |
| ------------------------------------------------ | -------------------------------------- |
| Handling missing data (imputation strategies)    | `notebooks/02`                         |
| Encoding categoricals (code-to-name mapping, one-hot) | `notebooks/03`, `src/load_encode.py` |
| Feature scaling (standardization)                | `notebooks/04`, `src/scale.py`         |
| Interaction and polynomial features              | `notebooks/05`, `src/features.py`      |
| Datetime feature extraction                      | `notebooks/06`, `src/features.py`      |
| Leak-free train/test splitting and model comparison | `notebooks/07`, `src/train_test.py` |

---

## 📈 Sample Finding

> RMSE dropped from 952.5 (raw table, no hour-of-day feature) to 250.2 after encoding and adding the hour, held at 249.9 after scaling, and fell to 221.5 after adding datetime features. Weekday demand peaks at the 8am and 5 to 6pm commute hours, while weekend demand is one broad midday hump.

![Average rides by hour of day, weekday vs weekend](outputs/figures/hourly_rides.png)

---

## 📄 Data Sources

- Demand dataset: [Bike Demand Dataset](https://www.kaggle.com/datasets/huseyincenik/bike-demand-capstone-project/data) by huseyincenik on Kaggle. Check the dataset's license on its Kaggle page before reusing the data.

### Data dictionary

| Column | Description |
|---|---|
| `timestamp` | Date and time of the hourly observation |
| `cnt` | Number of rides in that hour (the prediction target) |
| `t1` | Real temperature in Celsius |
| `t2` | "Feels like" temperature in Celsius |
| `hum` | Relative humidity (percent) |
| `wind_speed` | Wind speed (km/h) |
| `is_holiday` | 1 if the day is a holiday, otherwise 0 |
| `is_weekend` | 1 if the day is a Saturday or Sunday, otherwise 0 |
| `weather_code` | Category of weather conditions (codes below) |
| `season` | Meteorological season (codes below) |

### Engineered features

| Feature | How it is built |
|---|---|
| `hour` | Hour of day (0 to 23) from `timestamp` |
| `day_of_week` | Day of week (0 is Monday) |
| `month` | Month of year (1 to 12) |
| `is_rush_hour` | 1 for hours 7 to 9 and 16 to 19, otherwise 0 |
| `days_since_start` | Whole days since the first timestamp |
| `t_hum` | Temperature × humidity interaction |
| Polynomial terms | Squares and pairwise products of the numeric weather columns |

---

## 🔭 Possible Extensions

- Use a time-based split (train on earlier months, test on later ones) for an honest forecasting score
- Add lag features (rides in the previous hour/day) for a proper time-series angle
- Encode `hour` and `month` cyclically (sine/cosine) and test it with a linear model, where it matters more than for a forest
- Compare feature importances before/after engineering to see which engineered features
  the model actually leans on most
- Average scores over several random seeds or use cross-validation, since a single test split is noisy
