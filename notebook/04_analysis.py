from data_prep import prepare_data

import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

X_train_final, X_test_final, y_train, y_test = prepare_data()


model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(X_train_final, y_train)

gb_prob = model_gb.predict_proba(X_test_final)


# Пороговые значения сравниваются на тестовом наборе данных только для того, чтобы проиллюстрировать компромисс между точностью (precision) и полнотой (recall).
# Они не используются для настройки или выбора модели.
print('\n--- Сравнение Precision и Recall по порогам (Gradient Boosting) ---')
for threshold in [0.25, 0.3, 0.35, 0.4, 0.5]:
    pred_t = (gb_prob[:, 1] >=threshold).astype(int)
    rec_t = recall_score(y_test, pred_t)
    prec_t = precision_score(y_test, pred_t)
    f1_t = f1_score(y_test, pred_t)
    confm_t = confusion_matrix(y_test, pred_t)
    print(f'Threshold {threshold:.2f} | Precision: {prec_t:.4f} | Recall: {rec_t:.4f} | F1: {f1_t:.4f}')
    print(f'Confusion matrix: \n{confm_t}\n')


print('='*20, '\nFeature importance:')
# это оценка того, насколько сильно каждый признак (столбец в X) помог модели уменьшить ошибку при принятии решений
importances = pd.Series(
    model_gb.feature_importances_,
    index = X_train_final.columns
).sort_values(ascending = False)

print(importances.round(3))
print(f'Сумма важностей: {importances.sum():.4f}')
