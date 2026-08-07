# Hotel Booking Cancellation Prediction

A machine learning project that predicts whether a hotel booking will be cancelled, using classification algorithms trained on real booking data.

## Overview

Hotel cancellations are a major operational challenge. This project builds and compares multiple classification models to predict booking cancellations, helping hotels make data-driven decisions around overbooking strategies, resource allocation, and revenue management.

## Project Structure

```
Hotel_cancellation_prediction.ipynb   # Main notebook
booking.csv                           # Dataset
README.md
```

## Dataset

- Source file: `booking.csv`
- 36,285 booking records, 17 features
- No missing values or duplicate rows
- Target: `booking status` — `Canceled` or `Not_Canceled` (~67% not canceled)

### Features

| Feature | Type | Description |
|---|---|---|
| `number of adults` | int | Number of adults |
| `number of children` | int | Number of children |
| `number of weekend nights` | int | Weekend nights booked |
| `number of week nights` | int | Weekday nights booked |
| `type of meal` | categorical | Meal plan selected (4 categories) |
| `car parking space` | int | Parking space requested (0/1) |
| `room type` | categorical | Room type booked (7 types) |
| `lead time` | int | Days between booking and arrival |
| `market segment type` | categorical | Booking channel (5 segments) |
| `repeated` | int | Returning guest (0/1) |
| `P-C` | int | Number of previous cancellations |
| `P-not-C` | int | Number of previous non-cancelled bookings |
| `average price` | float | Average room price per night |
| `special requests` | int | Number of special requests (0–5) |
| `date of reservation` | date | Reservation date |

## Notebook Walkthrough

### 1. Data Loading & Inspection
- Load `booking.csv` and inspect shape, dtypes, and value distributions
- Confirm no nulls or duplicates across all 36,285 records

### 2. Exploratory Data Analysis (EDA)
- Statistical summaries for both numerical and categorical columns
- Count plots for categorical feature distributions (meal type, room type, market segment, booking status)
- Histograms and box plots for numerical features

### 3. Data Preprocessing
- Drop `Booking_ID` (unique identifier, not predictive) and `date of reservation`
- Separate target variable `booking status`
- One-hot encode categorical features
- Apply `StandardScaler` to numerical columns (zero mean, unit variance)

### 4. Feature Selection
- Use `SelectKBest` with `f_classif` to rank features by their statistical association with the target
- Select top **10** features for model training

### 5. Train/Test Split
- 80/20 stratified split → preserves class balance in both sets

### 6. Model Training
Four classifiers trained on the selected features:
- Logistic Regression (`liblinear` solver)
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN, k=5 baseline)

### 7. Hyperparameter Tuning
`GridSearchCV` with 5-fold cross-validation, scoring on ROC AUC:
- Decision Tree: tuned `max_depth`, `min_samples_split`
- Random Forest: tuned `n_estimators`, `max_depth`, `min_samples_split`
- KNN: tuned `n_neighbors` (3/5/7/9), `weights` (uniform/distance) → best: `n_neighbors=9`, `weights=distance`

### 8. Model Evaluation
Each model assessed with:
- Accuracy, Precision, Recall, F1-Score, ROC AUC
- Confusion matrix (heatmap)
- ROC curve

### 9. Model Comparison & Conclusion
Side-by-side comparison of all models with visualizations (accuracy bar chart, ROC curves overlay, feature importance for Random Forest).

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8057 | 0.7526 | 0.6064 | 0.6716 | 0.8670 |
| Decision Tree (Tuned) | 0.8691 | 0.8202 | 0.7691 | 0.7938 | 0.9260 |
| K-Nearest Neighbors (Tuned) | 0.8780 | 0.8370 | 0.7796 | 0.8073 | 0.9263 |
| **Random Forest (Tuned)** | **0.8891** | **0.8633** | **0.7860** | **0.8228** | **0.9470** |

The **tuned Random Forest** is the recommended model — highest accuracy, precision, F1, and ROC AUC across the board.

## Tech Stack

- Python 3
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, feature selection, models, tuning, evaluation

## Getting Started

1. Make sure `booking.csv` is in the same directory as the notebook.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook Hotel_cancellation_prediction.ipynb
   ```
4. Run all cells top to bottom. Each section is self-contained with markdown explanations.

## Key Takeaways

- **Lead time** and **special requests** are among the strongest predictors of cancellation
- Online bookings dominate the dataset (23k of 36k records), making market segment an important feature
- Ensemble methods (Random Forest) significantly outperform linear models for this task
- Hyperparameter tuning via GridSearchCV provided consistent improvements across all tree-based models
