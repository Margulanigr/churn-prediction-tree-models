"""
Hyperparameter tuning with 5-fold cross-validation on the training set only.
Slow: run it only when needed. The chosen values and results are in the README.
"""


from data_prep import prepare_data
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


X_train_final, X_test_final, y_train, y_test = prepare_data()


print('='*20, '\nDecision tree разные глубины:')

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

    scores_recall = cross_val_score(model, X_train_final, y_train, cv=5, scoring='recall', n_jobs = -1)
    scores_f1 = cross_val_score(model, X_train_final, y_train, cv=5, scoring='f1', n_jobs = -1)

    mean_score_r = scores_recall.mean()
    mean_score_f1 = scores_f1.mean()

    std_score_r = scores_recall.std()
    std_scores_f1 = scores_f1.std()

    print(f'Глубина {depth} | Средний recall: {mean_score_r:.4f} ± {std_score_r:.4f} | Средний f1: {mean_score_f1:.4f} ± {std_scores_f1:.4f}' )



print('='*20, '\ncheck depths with cross validation for random forest')
depths_rf_depth = [3,5,7,10,15, None]
for depth in depths_rf_depth:
    model = RandomForestClassifier(max_depth=depth, random_state=42)
    recall_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='recall', n_jobs = -1)
    f1_scores = cross_val_score(model, X_train_final, y_train, cv = 5, scoring='f1', n_jobs = -1)

    mean_recall = recall_scores.mean()
    mean_f1 = f1_scores.mean()

    std_recall = recall_scores.std()
    std_f1 = f1_scores.std()

    print(f'Глубина {depth} | Средний recall: {mean_recall:.4f} ± {std_recall:.4f} | Средний f1: {mean_f1:.4f} ± {std_f1:.4f}' )

print('-'*10, '\nmin samples_leaf')

rf_leaf = [1,3,5,10,20]
for leaf in rf_leaf:
    model = RandomForestClassifier(min_samples_leaf=leaf, max_depth=None, random_state=42)
    recall_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='recall', n_jobs = -1)
    f1_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='f1', n_jobs = -1)

    mean_recall = recall_scores.mean()
    mean_f1 = f1_scores.mean()

    std_recall = recall_scores.std()
    std_f1 = f1_scores.std()

    print(f'min_samples_leaf {leaf} | Средний recall: {mean_recall:.4f} ± {std_recall:.4f} | Средний f1: {mean_f1:.4f} ± {std_f1:.4f}')




#Подобрем параметры бустинга кросс-валидаицей на train
print('\n','='*30, '\ncross validation on boosting parameters for train ')
n_estimators = [50, 100, 200, 400]
for n in n_estimators:
    model = GradientBoostingClassifier(n_estimators=n, random_state=42)
    recall_scores = cross_val_score(model, X_train_final, y_train, cv = 5, scoring='recall', n_jobs = -1)
    f1_scores = cross_val_score(model, X_train_final, y_train, cv = 5, scoring = 'f1', n_jobs = -1)

    mean_score_r = recall_scores.mean()
    mean_score_f1 = f1_scores.mean()

    std_score_r = recall_scores.std()
    std_scores_f1 = f1_scores.std()

    print(f'n_estimators {n} | Средний recall: {mean_score_r:.4f} ± {std_score_r:.4f} | Средний f1: {mean_score_f1:.4f} ± {std_scores_f1:.4f}')

depth_gb = [2,3,4,5]
for depth in depth_gb:
    model = GradientBoostingClassifier(max_depth=depth, random_state=42)
    recall_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='recall', n_jobs = -1)
    f1_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='f1', n_jobs = -1)

    mean_score_r = recall_scores.mean()
    mean_score_f1 = f1_scores.mean()

    std_score_r = recall_scores.std()
    std_scores_f1 = f1_scores.std()

    print(f'max_depth {depth} | Средний recall: {mean_score_r:.4f} ± {std_score_r:.4f} | Средний f1: {mean_score_f1:.4f} ± {std_scores_f1:.4f}')
