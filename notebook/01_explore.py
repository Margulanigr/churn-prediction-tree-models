import pandas as pd



# расширяет ширину результатов в терминале
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv('../data/raw/Bank_Churn.csv')
print(df)

print('='*20)

print(df.head())

print('='*20)

print(df.iloc[3455:3461])

print('='*20)
# Для этого пригодятся функции pandas: read_csv, head, info, describe, isna().sum(), value_counts(normalize=True).
print(df.info())

print('='*20)

print(df.describe())

print('='*20)

print(df.isna().sum())

print('='*20)

print(df['churn'].value_counts(normalize=True)*100)

print('='*20)

print(df[['age', 'balance', 'estimated_salary', 'credit_score']].describe().T)

print('='*20)

print('Количество дубликатов:')
print(df.duplicated().sum())

print('='*20)

zero_balance_count = (df['balance']==0).sum()
print(f'Количество клиентов с нулевым балансом: {zero_balance_count}')

print('='*20)

churn_by_country = df.groupby('country')['churn'].mean()*100
print(f'Доля ушедших клиентов по странам (%): {churn_by_country}')

print('='*20)

country_stats = df.groupby('country')['churn'].agg(
    total_clients = 'count',
    churned_clients = 'sum',
    churn_rate_percent = lambda x: (x.mean() *100)
)
print(country_stats)

print('='*20)

churn_by_gender = df.groupby('gender')['churn'].mean()*100
print(f'Доля ушедших клиентов по гендерам (%): {churn_by_gender}')

print('='*20)

churn_by_product_number = df.groupby('products_number')['churn'].mean()*100
print(f'Доля ушедших клиентов по числу продуктов (%): {churn_by_product_number}')

print('='*20)

churn_by_active_member = df.groupby('active_member')['churn'].mean()*100
print(f'Доля ушедших по активным участникам (%): {churn_by_active_member}')


