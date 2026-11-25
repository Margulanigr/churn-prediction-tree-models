from data_prep import prepare_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X_train_final, X_test_final, y_train, y_test = prepare_data()

model = DecisionTreeClassifier(max_depth=3, random_state=42)

model = model.fit(X_train_final, y_train)

pred = model.predict(X_test_final)

accuracy = accuracy_score(y_test, pred)

print('Predict: ', pred)

print(f'Точность модели (Accuracy) (depth = 3): {accuracy}')

churned_count = pred.sum()
print(f"Модель предсказала уход для {churned_count} клиентов (из {len(pred)} в тестовой выборке)")

conf = confusion_matrix(y_test, pred)
print(f'Матрица ошибок (Confusion Matrix) (depth = 3): \n{conf}')

class_report = classification_report(y_test, pred)
print(f'classification_report:(depth = 3)\n{class_report}')


# Создаем «пустышку», которая всегда выбирает самый частый класс
dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)

dummy_model.fit(X_train_final, y_train)

dummy_pred = dummy_model.predict(X_test_final)

dummy_class_rep = classification_report(y_test, dummy_pred, zero_division=0)

print(f'метрики Dummy-модели: \n{dummy_class_rep}')


print('\n','-'*20, '\nDecision tree(max_depth = 7)')
model = DecisionTreeClassifier(max_depth=7, random_state=42)
model.fit(X_train_final,y_train)
pred_test_d7 = model.predict(X_test_final)
pred_prob = model.predict_proba(X_test_final)
class_rep = classification_report(y_test, pred_test_d7)
conf_matr = confusion_matrix(y_test, pred_test_d7)
print(f'Classification report(max_depth = 7):\n{class_rep}')
print(f'Confusion matrix (max_depth = 7):\n{conf_matr}')

roc_auc = roc_auc_score(y_test, pred_prob[:,1])
print(f'ROC AUC: {roc_auc:.4f}')

print('='*20, '\nRandom Forest:')

#Random Forest
modelrf = RandomForestClassifier(random_state=42)
modelrf.fit(X_train_final, y_train)
pred_rf = modelrf.predict(X_test_final)
rf_report = classification_report(y_test, pred_rf)
rf_conf = confusion_matrix(y_test, pred_rf)
print(f'Classification report:\n{rf_report}')
print(f'Confusion matrix: \n{rf_conf}')

rf_prob = modelrf.predict_proba(X_test_final)
rf_roc_auc = roc_auc_score(y_test, rf_prob[:,1])
print(f'ROC AUC(random forest): {rf_roc_auc:.4f}')



print('-'*10, '\nпроверяем, лес переобучается ли или нет(w train):')

pred_rf_train = modelrf.predict(X_train_final)
accuracy_for_train = accuracy_score(y_train, pred_rf_train)
recall_for_train = recall_score(y_train, pred_rf_train)
print(f'Accuracy for train: {accuracy_for_train}')
print(f'Recall for train: {recall_for_train}')


print('\n','='*30, '\nGradient Boosting:')
model_gb = GradientBoostingClassifier(random_state=42)
model_gb.fit(X_train_final, y_train)
pred_gb = model_gb.predict(X_test_final)
pred_gb_train = model_gb.predict(X_train_final)

accuracy_gb = accuracy_score(y_test, pred_gb)
report_gb = classification_report(y_test, pred_gb)
confm_gb = confusion_matrix(y_test, pred_gb)
accuracy_gb_train = accuracy_score(y_train, pred_gb_train)
recall_gb_train = recall_score(y_train, pred_gb_train)

print(f'Classification_report(gradient boosting): \n{report_gb}'
      f'\nConfusion Matrix: \n{confm_gb}'
      f'\nAccuracy for test: \n{accuracy_gb}'
      f'\nAccuracy for train (gradient boosting): {accuracy_gb_train:.4f}'
      f'\nRecall for train (gradient boosting): {recall_gb_train:.4f}')

gb_prob = model_gb.predict_proba(X_test_final)
roc_auc_gb = roc_auc_score(y_test, gb_prob[:,1])
print(f'ROC-AUC (gradient boosting): {roc_auc_gb:.4f}')

