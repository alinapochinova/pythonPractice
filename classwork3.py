import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_clients = 1000

def generate_non_negative(mean, std, size):
    data = np.random.normal(mean, std, size)
    while np.any(data < 0):
        negative_indices = data < 0
        new_values = np.random.normal(mean, std, np.sum(negative_indices))
        data[negative_indices] = new_values
    return data

products = generate_non_negative(20000, 5000, n_clients)
entertainment = generate_non_negative(10000, 4000, n_clients)
online_shopping = generate_non_negative(15000, 7000, n_clients)

expenses_matrix = np.column_stack([products, entertainment, online_shopping])

total_expenses_per_client = np.sum(expenses_matrix, axis=1)

correlation_matrix = np.corrcoef(expenses_matrix, rowvar=False)

threshold_95 = np.percentile(total_expenses_per_client, 95)
top_5_percent_clients = total_expenses_per_client[total_expenses_per_client >= threshold_95]

total_all_expenses = np.sum(total_expenses_per_client)
top_5_expenses = np.sum(top_5_percent_clients)
percentage_top_5 = (top_5_expenses / total_all_expenses) * 100

plt.figure(figsize=(8, 6))
categories = ['Продукты', 'Развлечения', 'Онлайн-покупки']
box_data = [products, entertainment, online_shopping]
plt.boxplot(box_data, labels=categories)
plt.title('Распределение расходов по категориям')
plt.ylabel('Расходы, руб.')
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(products, online_shopping, alpha=0.6, s=20)
plt.xlabel('Расходы на продукты, руб.')
plt.ylabel('Расходы на онлайн-покупки, руб.')
plt.title('Продукты vs Онлайн-покупки')
plt.grid(True, alpha=0.3)
plt.show()

print("Общие расходы каждого клиента:")
print(total_expenses_per_client.astype(int))

print("\nКорреляция между категориями:")
print(correlation_matrix)

print(f"\nТоп-5% клиентов формируют {percentage_top_5:.1f}% общих расходов")