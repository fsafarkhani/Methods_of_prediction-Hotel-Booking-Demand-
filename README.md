# Hotel Booking Cancellation Prediction (Deep Learning)

Predicting hotel booking cancellations using a Kaggle dataset and Keras NNs.

## 📂 Project Overview
- Goal: Predict `is_canceled` to help revenue management & overbooking.
- Data: Kaggle – Hotel Booking Demand (119,390 rows, 30 features).
- Methods: One-Hot Encoding, Scaling, SMOTE; 10 NN variants with Dropout & BatchNorm.
- Outcome: Best model (3-layer NN) → Val Acc 86.8%, Test Acc 87.3%, F1 83.1%.

## 🗃️ Dataset
- Source: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Target: `is_canceled` (0/1)

## 🧭 Approach
1) Cleaning (fix NA, drop invalid rows, remove `company`)  
2) EDA (lead time distribution, correlation heatmap)  
3) Preprocess (ColumnTransformer: scale num + OHE cat)  
4) Balance (SMOTE)  
5) Modeling (10 NN configs; early stopping)  
6) Evaluation (Accuracy, Precision, Recall, F1; Confusion Matrix)

## 🥇 Results
- **Best:** Model-8 (256-128-64 + Dropout 0.1 + BatchNorm)  
- **Test:** Acc 87.3% • Precision 82.1% • Recall 84.1% • F1 83.1%  
- **Insights:** Lead time strongly drives cancellations; ADR has weak correlation.
