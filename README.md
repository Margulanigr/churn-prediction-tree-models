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
- Clients from Germany: the churn rate is 32.4%, while France and Spain make up around 16,5%
- Churn rate among women is 25.1% vs 16.5% among men
- Inactive clients have a churn rate of 26.9%, compared to a 14.3% for active clients
- For number of products, the churn rate is 7.6% for 2 products, 27.7% for 1 product. For 3 it's reaches 82.7% and for 4 it is 100%, but there are only 326 clients with 3 or 4 number of products, therefore, the conclusion is a cautious one

## Models and results
| Model| Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Dummy (most frequent)	| 0.00 | 0.00 | 0.00 | 0.50 |
| Decision Tree, depth 3 | 0.87 | 0.24 | 0.37 | not calculated |
| Decision Tree, depth 7 | 0.68 | 0.51 | 0.59 | 0.83 |

## How to run
```
pip install pandas scikit-learn
Run notebook/01_explore.py, then notebook/02_prepare.py
```

## Next steps
Random Forest, Gradient Boosting, SQL analysis

