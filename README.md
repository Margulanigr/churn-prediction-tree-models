# churn-prediction-tree-models
Predicting bank customer churn with Decision Tree, Random Forest and Gradient Boosting (python, scikit-learn)

## Dataset
- Source: [https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)
- 10,000 clients, 12 columns, target: churn(1 = left the bank)
- The data is not stored in the repo: download it and put it into data/raw/Bank_Churn.csv


## Project Structure
```
notebook/01_explore.py   - data exploration
notebook/02_prepare.py   - preprocessing, models, evaluation
```

## Key findings from EDA
- 20.4% of clients left the bank
- Clients from Germany: the churn rate is 32.4%, while about 16% in France and Spain 
- Churn rate among women is 25.1% vs 16.5% among men
- Inactive clients have a churn rate of 26.9%, compared to a 14.3% for active clients
- For number of products, the churn rate is 7.6% for 2 products, 27.7% for 1 product. For 3 products it's reaches 82.7% and for 4 it is 100%, but there are only 326 clients with 3 or 4 number of products, therefore, the conclusion is a cautious one

## Models and results

| Model                                  | Precision | Recall | F1   | ROC-AUC |
|----------------------------------------|---|---|------|---|
| Dummy (most frequent)	                 | 0.00 | 0.00 | 0.00 | 0.50 |
| Decision Tree, depth 3                 | 0.87 | 0.24 | 0.37 | not calculated |
| Decision Tree, depth 7                 | 0.68 | 0.51 | 0.59 | 0.83 |
| Random Forest (default parameters)     | 0.78 | 0.46 | 0.58 | 0.85 |
| Gradient Boosting (default parameters) | 0.79 | 0.49 | 0.60 | 0.87 |

- Random Forest hyperparameters were checked with 5-fold cross-validation on the training set (max_depth: 3-15 and None; min_samples_leaf: 1:20). No improvement beyond the cross_validation noise was found, so the default parameters were kept.
- It has a higher ROC-AUC (0.85 vs 0.83), so it ranks clients by churn risk better.
- Its F1-score is virtually the same as the depth-7 decision tree (0.58 vs 0.59).
- The forest overfits the training data (accuracy and recall are 1.00 on train vs 0.86 and 0.46 on test), but its ROC_AUC on test is good (0.85), and limiting tree depth did not help.
- Gradient Boosting shows no sign of overfitting with default parameters: accuracy is 0.87 on both train and test, recall is 0.50 on train vs 0.49 on test. It builds small trees one by one, each correcting the errors of the previous one.
- It has the highest ROC-AUC among the models (0.87).
- Gradient Boosting parameters (n_estimators 50-400, max_depth 2-5) were checked with 5-fold cross_validation on the training set. No improvement beyond the cross-validation noise was found, so the default parameters (100 trees, depth 3) were kept.

## How to run
```
pip install pandas scikit-learn
Run notebook/01_explore.py, then notebook/02_prepare.py
```

## Next steps
Feature importance, error analysis, SQL analysis