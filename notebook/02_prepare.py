import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('../data/raw/Bank_Churn.csv')

X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

#Кодирвание категорий через OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Кодируем сразу оба столбца
encoded_array_train = encoder.fit_transform(X_train[['country', 'gender']])
encoded_array_test = encoder.transform(X_test[['country', 'gender']])

# Получаем новые названия столбцов
encoded_columns = encoder.get_feature_names_out(['country', 'gender'])

# Превращаем обратно в DataFrame и соединяем с остальными данными
encoded_df_train = pd.DataFrame(encoded_array_train, columns=encoded_columns, index=X_train.index)
encoded_df_test = pd.DataFrame(encoded_array_test, columns = encoded_columns, index = X_test.index)


# Собираем итоговый датасет (исключая старые текстовые столбцы)
X_train_final = pd.concat([X_train.drop(['country', 'gender'], axis = 1), encoded_df_train],axis=1)
X_test_final = pd.concat([X_test.drop(['country', 'gender'], axis=1), encoded_df_test], axis=1)

# print(f'X_train: {X_train_final.shape}')
# print(f'X_test: {X_test_final.shape}')
#
# print('='*20)
#
# print(X_train_final.isna().sum().sum())
# print(X_test_final.isna().sum().sum())
# print('='*20)
#
# print(X_train_final.dtypes)
#
# print('='*20)
#
# print(list(X_train_final.columns)==list(X_test_final.columns))

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

print('='*20, '\nразные глубины:')

#проверим результаты модели при разных значений глубины (max_depth)
depths = [1,2,3,5,7,10,None]
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model = model.fit(X_train_final, y_train)

    pred_train = model.predict(X_train_final)
    pred_test = model.predict(X_test_final)

    recall_train = recall_score(y_train, pred_train)
    recall_test = recall_score(y_test, pred_test)

    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)


    print(f'Полнота модели (Recall) for train (depth:{depth}): {recall_train:.3f}')

    print(f'Полнота модели (Recall) for test (depth:{depth}): {recall_test:.3f}')

    print(f'Точность модели (Accuracy) for train (depth:{depth}): {acc_train:.3f}')

    print(f'Точность модели (Accuracy) for test (depth: {depth}): {acc_test:.3f}')

    print('-'*10)


print('\ncross validation:')
# Подбор оптимальной глубины дерева (max_depth) на обучающей выборке с помощью 5-фолдовой кросс-валидации.
# Для каждой глубины считаем среднее значение (mean) и стабильность/разброс (std) метрик Recall и F1
# по 5 разбиениям, чтобы выбрать баланс между умением находить уходящих клиентов и защитой от переобучения.
for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    scores_recall = cross_val_score(model, X_train_final, y_train, cv=5, scoring='recall')
    scores_f1 = cross_val_score(model, X_train_final, y_train, cv=5, scoring='f1')

    mean_score_r = scores_recall.mean()
    mean_score_f1 = scores_f1.mean()

    std_score_r = scores_recall.std()
    std_scores_f1 = scores_f1.std()

    print(f'Глубина {depth} | Средний recall: {mean_score_r:.4f} ± {std_score_r:.4f} | Средний f1: {mean_score_f1:.4f} ± {std_scores_f1:.4f}' )

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

# print('='*20, '\ncheck depths with cross validation for random forest')
# depths_rf_depth = [3,5,7,10,15, None]
# for depth in depths_rf_depth:
#     model = RandomForestClassifier(max_depth=depth, random_state=42)
#     recall_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='recall')
#     f1_scores = cross_val_score(model, X_train_final, y_train, cv = 5, scoring='f1')
#
#     mean_recall = recall_scores.mean()
#     mean_f1 = f1_scores.mean()
#
#     std_recall = recall_scores.std()
#     std_f1 = f1_scores.std()
#
#     print(f'Глубина {depth} | Средний recall: {mean_recall:.4f} ± {std_recall:.4f} | Средний f1: {mean_f1:.4f} ± {std_f1:.4f}' )
#
# print('-'*10, '\nmin samples_leaf')
#
# rf_leaf = [1,3,5,10,20]
# for leaf in rf_leaf:
#     model = RandomForestClassifier(min_samples_leaf=leaf, max_depth=None, random_state=42)
#     recall_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='recall')
#     f1_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='f1')
#
#     mean_recall = recall_scores.mean()
#     mean_f1 = f1_scores.mean()
#
#     std_recall = recall_scores.std()
#     std_f1 = f1_scores.std()
#
#     print(f'min_samples_leaf {leaf} | Средний recall: {mean_recall:.4f} ± {std_recall:.4f} | Средний f1: {mean_f1:.4f} ± {std_f1:.4f}')
#


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
      f'\nAccuracy for test: \b{accuracy_gb}'
      f'\nAccuracy for train (gradient boosting): {accuracy_gb_train:.4f}'
      f'\nRecall for train (gradient boosting: {recall_gb_train:.4f}')

gb_prob = model_gb.predict_proba(X_test_final)
roc_auc_gb = roc_auc_score(y_test, gb_prob[:,1])
print(f'ROC-AUC (gradient boosting): {roc_auc_gb:.4f}')



