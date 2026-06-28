# E-Commerce Sales Dataset — EDA Guide # EDA-Project-Firdaus-Khan-Intern-ID-CITS2712

## Dataset Overview
**File:** `ecommerce_sales.csv`  
**Rows:** 35 | **Columns:** 15  
Simulated Indian e-commerce orders across 5 cities, Jan–Mar 2024.

---

## Column Reference

| Column | Type | Description |
|---|---|---|
| order_id | int | Unique order identifier |
| customer_id | str | Customer code |
| age | int | Customer age |
| gender | str | Male / Female |
| city | str | Mumbai, Delhi, Bangalore, Pune, Hyderabad, Chennai |
| category | str | Product category |
| product | str | Product name |
| quantity | int | Units ordered |
| unit_price | float | Price per unit (₹) |
| discount_pct | float | Discount applied (%) |
| payment_method | str | UPI, Credit Card, Debit Card, Cash, EMI |
| order_date | date | Order placed date |
| delivery_days | int | Days taken to deliver |
| rating | float | Customer rating (1–5) |
| returned | str | Yes / No |

---

## EDA Steps to Perform

### 1. Load & Inspect
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ecommerce_sales.csv')
df['order_date'] = pd.to_datetime(df['order_date'])

df.shape
df.head()
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### 2. Feature Engineering
```python
df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount_pct']/100)
df['month'] = df['order_date'].dt.month_name()
df['returned_flag'] = (df['returned'] == 'Yes').astype(int)
```

### 3. Univariate Analysis
```python
# Distribution of age
sns.histplot(df['age'], kde=True)

# Category frequency
df['category'].value_counts().plot(kind='bar')

# Rating distribution
sns.boxplot(x=df['rating'])
```

### 4. Bivariate Analysis
```python
# Revenue by category
df.groupby('category')['revenue'].sum().sort_values().plot(kind='barh')

# Rating vs delivery days
sns.scatterplot(x='delivery_days', y='rating', hue='returned', data=df)

# Payment method vs revenue
sns.boxplot(x='payment_method', y='revenue', data=df)
```

### 5. Correlation Heatmap
```python
sns.heatmap(df[['age','quantity','unit_price','discount_pct','delivery_days','rating','revenue']].corr(), annot=True, cmap='coolwarm')
```

### 6. Categorical Insights
```python
# Return rate by category
df.groupby('category')['returned_flag'].mean().sort_values(ascending=False)

# Revenue by city
df.groupby('city')['revenue'].sum().sort_values(ascending=False)

# Gender split
df['gender'].value_counts(normalize=True) * 100
```

### 7. Time-Series Trend
```python
df.groupby('order_date')['revenue'].sum().plot(title='Daily Revenue Trend')
```

---

## Key Questions to Answer via EDA
1. Which category generates the highest revenue?
2. Does higher discount lead to more returns?
3. Which city has the most orders?
4. Is delivery time affecting ratings?
5. Which payment method is most preferred?
