import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('../data/raw/Bank_Churn.csv')

X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)


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

print(f'X_train: {X_train_final.shape}')
print(f'X_test: {X_test_final.shape}')

print('='*20)

print(X_train_final.isna().sum().sum())
print(X_test_final.isna().sum().sum())
print('='*20)

print(X_train_final.dtypes)

print('='*20)

print(list(X_train_final.columns)==list(X_test_final.columns))

