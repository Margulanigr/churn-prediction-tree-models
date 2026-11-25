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
- Age: churn grows sharply with age, from 7.5% (up to 30) and 12.1% (31-40) to 34.0% (41-50) and 56.2% (51-60), then drops to 24.8% for clients over 60 (464 clients).
- Balance: clients with zero balance churn less (13.8% vs 24.1%). However, there are no zero-balance clients in Germany, which has the highest churn. Within France and Spain the gap is smaller (13.9% vs 18.2% and 13.6% vs 19.6%), so a large part of the overall gap is a country effect.

## Models and results

| Model                                  | Precision | Recall | F1   | ROC-AUC |
|----------------------------------------|---|---|------|---|
| Dummy (most frequent)	                 | 0.00 | 0.00 | 0.00 | 0.50 |
| Decision Tree, depth 3                 | 0.87 | 0.24 | 0.37 | not calculated |
| Decision Tree, depth 7                 | 0.68 | 0.51 | 0.59 | 0.83 |
| Random Forest (default parameters)     | 0.78 | 0.46 | 0.58 | 0.85 |
| Gradient Boosting (default parameters) | 0.79 | 0.49 | 0.60 | 0.87 |

Decision Tree
- The depth (7) was chosen with 5-fold cross-validation on the training set; the test set was used only once for the final check.
- A tree without a depth limit overfits: accuracy is 1.00 on train vs 0.78 on test.

Random Forest
- max_depth (3-15 and None) and min_samples_leaf (1-20) were checked with 5-fold cross-validation on the training set. No improvement beyond the cross-validation noise was found, so the default parameters were kept.
- The forest overfits the training data (accuracy and recall are 1.00 on train vs 0.86 and 0.46 on test), but its ROC-AUC on test is good (0.85), and limiting tree depth did not help.
- ROC-AUC is higher than the depth-7 tree (0.85 vs 0.83), so it ranks clients by churn risk better. F1 is virtually the same (0.58 vs 0.59).

Gradient Boosting
- n_estimators (50-400) and max_depth (2-5) were checked with 5-fold cross-validation on the training set. No improvement beyond the noise was found, so the default parameters (100 trees, depth 3) were kept.
- No sign of overfitting: accuracy is 0.87 on both train and test, recall is 0.50 on train vs 0.49 on test.
- It has the highest ROC-AUC among the models (0.87).

## Feature importance
| Feature          | Importance |
|------------------|------------|
| age              | 0.388      |
| products_number  | 0.300      |
| active_member    | 0.114      |
| balance          | 0.089      |
| country_Germany  | 0.056      |

- The top features match the EDA: number of products, activity and Germany all showed large churn differences. Age was not checked in the first EDA, and the follow-up analysis confirmed it (churn from 7.5% to 56.2% across age groups).
- Salary, credit score, tenure and having a credit card have almost no importance.
- Gender has low importance (0.013) despite the raw gap in churn (25.1% vs 16.5%): once the other features are known, it adds little information.
- Balance is partly a country effect (see EDA), so its importance should not be read as a direct effect of balance.
- Importance shows how much the model uses a feature, not the direction of the effect or its cause.

## How to run
```
pip install pandas scikit-learn
Run notebook/01_explore.py, then notebook/02_prepare.py
```

## Next steps
Decision threshold analysis, error analysis, SQL analysis