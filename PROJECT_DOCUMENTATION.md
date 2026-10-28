# PROJECT DOCUMENTATION
## Online Transaction Fraud Detection System Using Explainable Machine Learning

**Course:** Forensic Analytics  
**Subject Code:** CSE4074  
**Institution:** Vellore Institute of Technology, Chennai  
**Semester:** Fall 2026-27  
**Professor:** Dr. S. Johanan Joysingh  
**Developed by:** Hannah Cinderella L, Kishan V

---

## 1. Introduction

Financial fraud detection is an important application of data analytics and machine learning. Financial transaction systems process large volumes of transactions, making manual inspection of every transaction difficult. Automated fraud detection models can examine transaction characteristics and identify patterns associated with potentially fraudulent activity.

This project develops an Online Transaction Fraud Detection System using Python, machine learning, and Streamlit. It provides a dashboard for checking transactions, displaying fraud probabilities, monitoring simulated transactions, and interpreting individual predictions using LIME.

## 2. Problem Statement

Fraud detection is challenging because genuine transactions greatly outnumber fraudulent transactions. A model can achieve high overall accuracy simply by predicting most transactions as genuine, even when it fails to detect fraud.

The project investigates machine learning models for identifying fraud, applies class-balancing techniques, evaluates fraud-specific performance, and presents predictions through an interactive dashboard.

## 3. Objectives

The main objectives are:

1. Preprocess transaction data for machine learning.
2. Address class imbalance using SMOTE.
3. Train and compare multiple classification algorithms.
4. Evaluate the models using fraud-specific performance metrics.
5. Build an interactive Streamlit dashboard.
6. Display fraud probabilities and risk classifications.
7. Explain individual predictions using LIME.

## 4. Dataset Description

The training notebook uses the PaySim synthetic financial transaction dataset. The notebook contains 6,362,620 records.

The target variable is `isFraud`:

- `0`: Genuine transaction
- `1`: Fraudulent transaction

The model uses ten features: transaction step, amount, sender balance before and after the transaction, receiver balance before and after the transaction, and binary indicators for CASH_OUT, DEBIT, PAYMENT, and TRANSFER.

The notebook removes `nameOrig`, `nameDest`, and `isFlaggedFraud` from the model inputs.

Because PaySim is synthetic, results on this dataset do not guarantee identical performance on real financial transactions.

## 5. Proposed Methodology

The project workflow consists of the following stages.

### 5.1 Data preprocessing

The transaction type is converted into binary indicator columns. The input variables and fraud labels are separated, and the dataset is split into training and test sets using stratified sampling with an 80:20 ratio.

### 5.2 Class imbalance handling

SMOTE is applied to the training data to increase representation of the minority fraud class. The notebook records 5,083,526 genuine and 6,570 fraudulent training cases before SMOTE, followed by 5,083,526 examples of each class after resampling.

The test set is not resampled.

### 5.3 Feature scaling

`StandardScaler` is fitted on the resampled training features. The same fitted scaler is used to transform the test features and must be used by the application.

### 5.4 Model training

The notebook compares four algorithms:

- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost

Each model is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

### 5.5 Model selection

The notebook calculates a ranking score using:

`Rank Score = 0.7 × Recall + 0.3 × F1-score`

This gives greater importance to fraud recall while still considering the balance between precision and recall.

In the original model comparison, the Decision Tree was selected with a ranking score of approximately 0.9176.

### 5.6 Calibration experiment

An additional experiment applies isotonic calibration to a Decision Tree. The purpose of calibration is to improve the relationship between predicted probabilities and observed outcomes.

However, the current isotonic model evaluation predicts every test transaction as genuine at a 0.5 threshold. This experiment therefore requires further investigation and must not be described as a successful improvement in fraud detection.

## 6. Model Evaluation and Results

The original Decision Tree results recorded in the notebook are:

| Metric | Result |
|---|---:|
| Accuracy | 99.94% |
| Fraud precision | 70.93% |
| Fraud recall | 96.10% |
| Fraud F1-score | 81.62% |
| ROC-AUC | 98.03% |
| Average precision | 0.6818 |

The original Decision Tree confusion matrix is:

| | Predicted genuine | Predicted fraud |
|---|---:|---:|
| Actual genuine | 1,270,234 | 647 |
| Actual fraud | 64 | 1,579 |

These results show that the original Decision Tree correctly detected 1,579 of the 1,643 fraudulent transactions in the held-out test set, while incorrectly flagging 647 genuine transactions.

### 6.1 Isotonic model evaluation

The current isotonic model evaluation produced the following confusion matrix:

| | Predicted genuine | Predicted fraud |
|---|---:|---:|
| Actual genuine | 1,270,881 | 0 |
| Actual fraud | 1,643 | 0 |

At the 0.5 threshold, the isotonic model achieved 0% fraud recall. Its overall accuracy appears high because genuine transactions dominate the test dataset, but it failed to detect any fraud cases.

Therefore, the original Decision Tree's performance figures must not be attributed to the isotonic model. The final application should use a model whose compatibility and evaluation results have been verified.

## 7. System Architecture

The application follows this workflow:

**Transaction input → Feature construction → Feature scaling → Trained model → Fraud probability → Risk classification → Dashboard and explanation**

The system loads the saved model, scaler, feature names, and LIME background data. Transaction details are converted into the same ten-feature order used during training.

The model estimates the probability of fraud. The application then displays the probability and assigns a risk category using configured thresholds.

## 8. Streamlit Application

The application title is **Online Transaction Fraud Detection System**, with the caption *Predictive Analytics with Explainable Machine Learning*.

### 8.1 Real-Time Monitoring

The monitoring mode generates simulated transactions and displays:

- Transaction ID, amount, and type
- Fraud probability and risk score
- Predicted risk classification
- Recent transaction history
- Total monitored transactions and predicted fraud count
- Fraud probability trend and rolling average

The monitoring interface includes start, pause, and resume controls.

The generated transactions are simulated examples. A generated scenario described as fraudulent is not a verified fraud case, and the dashboard's predicted fraud count is not a measure of actual fraud detection accuracy.

### 8.2 Manual Transaction Check

Users enter a transaction step, amount, sender balances, receiver balances, and transaction type. The application builds the model input, scales the features, obtains the model's fraud probability, and displays a prediction.

The manual check is useful for demonstrating how the model responds to different transaction characteristics.

### 8.3 Risk Explanation

The application includes rule-based indicators for large transaction amounts, sender-balance mismatches, selected transaction types, and receiver-balance changes.

These indicators are separate from the machine learning prediction. They provide additional context but do not prove that a transaction is fraudulent.

## 9. Explainable AI Using LIME

LIME stands for Local Interpretable Model-Agnostic Explanations. It approximates the model's behavior around an individual input to provide a local explanation.

In this application, LIME perturbs the transaction features and requests predictions from the saved model. It then returns feature contributions that are displayed in a table and bar chart.

These explanations can help users understand how features influence a particular prediction. They are local approximations rather than causal explanations or proof of fraud.

## 10. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn and SMOTE
- Streamlit
- LIME
- Joblib
- Matplotlib and Seaborn
- Jupyter Notebook / Kaggle

## 11. Project Files

| File | Purpose |
|---|---|
| `app.py` | Implements the Streamlit dashboard and prediction workflow |
| `fa-proj.ipynb` | Contains data preprocessing, model training, comparison, and evaluation |
| `fraud_model_isotonic.pkl` | Isotonic-calibrated model currently referenced by the application |
| `scaler.pkl` | Feature scaling object |
| `feature_names.pkl` | Names of the input features |
| `explain_background.pkl` | Background data for LIME explanations |
| `requirements.txt` | Lists the Python dependencies |

The saved model, scaler, feature order, and LIME data must be compatible with the same training pipeline.

## 12. Limitations

1. The dataset is synthetic and may not fully reflect real-world fraud.
2. Genuine transactions greatly outnumber fraudulent transactions.
3. Overall accuracy alone is insufficient to evaluate fraud detection.
4. Simulated monitoring data does not have verified ground-truth labels.
5. The isotonic calibration experiment currently fails to detect fraud at the 0.5 threshold.
6. LIME explanations are approximate and should not be interpreted as causal proof.

## 13. Future Enhancements

- Investigate and validate the probability calibration process.
- Select a decision threshold using validation data.
- Evaluate the final model on an untouched test set.
- Add a testing mode for real labelled dataset examples.
- Improve logging and monitoring of false positives and false negatives.
- Explore deployment and integration with financial transaction systems.

## 14. Conclusion

This project demonstrates an end-to-end machine learning workflow for transaction fraud detection, from data preprocessing and class balancing to model comparison, evaluation, and interactive application development.

The Streamlit dashboard demonstrates manual transaction checking, simulated monitoring, and LIME-based local explanations. The model evaluation also highlights why fraud recall, precision, and calibration need to be checked carefully instead of relying only on overall accuracy.

Before final deployment, the model and scaler must be verified together, and the selected model's fraud detection performance must be confirmed on held-out labelled data.

## 15. Team Members

- Hannah Cinderella - 23MIA1043
- Kishan V - 23MIA1138

## 16. Acknowledgement

This project was developed as part of the Forensic Analytics course (CSE4074) at Vellore Institute of Technology, Chennai, under the guidance of Dr. S. Johanan Joysingh.