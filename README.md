# hotel-booking-cancellation-prediction
Machine Learning project for predicting hotel booking cancellations using classification algorithms.
# Hotel Booking Cancellation Prediction Using Machine Learning

## Project Overview

This project predicts whether a hotel booking will be canceled or not using machine learning classification algorithms. Four different models are trained, tuned, and compared to determine the best-performing algorithm.

---

## Dataset

- **Dataset Name:** Hotel Booking Cancellation Prediction
- **Source:** Kaggle
- **Machine Learning Type:** Supervised Learning
- **Problem Type:** Binary Classification
- **Target Variable:** `booking_status`

---

## Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

---

## Data Preprocessing

The following preprocessing steps were performed:

- Data cleaning
- Handling missing values
- Label Encoding
- Feature Scaling using StandardScaler
- Train-Test Split (80% Training, 20% Testing)

---

## Hyperparameter Tuning

The following models were optimized using GridSearchCV:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

Logistic Regression was used as the baseline model.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC Score
- Confusion Matrix

---

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## How to Run

1. Install the required libraries:

```
pip install -r requirements.txt
```

2. Open the notebook:

```
Hotel_Booking_Cancellation_Prediction.ipynb
```

3. Run all cells from top to bottom.

---

## Project Files

- `Hotel_Booking_Cancellation_Prediction.ipynb`
- `requirements.txt`
- `README.md`
- `Hotel_Booking_Cancellation_Presentation.pptx`

---

## Author
- **Lecturer** Mr. HIM Soklong
- **Name:** Sothea Rach, Thun MengThean, Srey Tethtebveboth
- **Course:** Machine Learning
- **University:** Phnom Penh International University
