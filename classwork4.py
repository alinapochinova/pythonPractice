import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns

print("Генерация датасета carsharing.csv...")

np.random.seed(42)

num_days = 180
num_cars = 20

records_per_day = np.random.randint(5, 16, size=num_days)

dates = pd.date_range('2024-01-01', periods=num_days, freq='D')

data = []

for i, date in enumerate(dates):
    for _ in range(records_per_day[i]):
        car_id = np.random.randint(1, num_cars + 1)
        duration = np.random.randint(15, 181)
        base_distance = duration / 60 * 20
        distance = max(1, base_distance + np.random.normal(0, 3))
        revenue_per_km = np.random.uniform(8, 12)
        revenue = round(distance * revenue_per_km, 2)
        fuel_cost_per_km = np.random.uniform(2, 4)
        fuel_cost = round(distance * fuel_cost_per_km, 2)
        data.append([date, car_id, duration, round(distance, 2), revenue, fuel_cost])

df = pd.DataFrame(data, columns=["Date", "Car_ID", "Duration_min", "Distance_km", "Revenue", "Fuel_cost"])

df.to_csv("carsharing.csv", index=False, encoding="utf-8")

print("Файл carsharing.csv сгенерирован! Количество строк:", df.shape[0])
print(df.head())

data = pd.read_csv("carsharing.csv", parse_dates=["Date"])

data["Profit"] = data["Revenue"] - data["Fuel_cost"]
data["Avg_speed"] = data["Distance_km"] / (data["Duration_min"] / 60)

print("\n" + "=" * 50)
print("АНАЛИЗ ДАННЫХ КАРШЕРИНГА")
print("=" * 50)

print("\n1. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ")

total_revenue = data.groupby("Car_ID")["Revenue"].sum()
top_cars = total_revenue.nlargest(5)

avg_profitability = data.groupby("Car_ID")["Profit"].mean()
low_profit_cars = avg_profitability.nsmallest(5)

print("Топ-5 автомобилей по выручке:")
for car_id, revenue in top_cars.items():
    print(f"Автомобиль {car_id}: {revenue:.2f} руб.")

print("\n5 автомобилей с наименьшей рентабельностью:")
for car_id, profit in low_profit_cars.items():
    print(f"Автомобиль {car_id}: {profit:.2f} руб. (средняя прибыль)")

print("\n2. АНАЛИЗ ВРЕМЕННЫХ РЯДОВ")

daily_revenue = data.groupby("Date")["Revenue"].mean()

rolling_avg = daily_revenue.rolling(window=7).mean()

plt.figure(figsize=(12, 6))
plt.plot(daily_revenue.index, daily_revenue.values, label="Средняя выручка за день", alpha=0.7)
plt.plot(rolling_avg.index, rolling_avg.values, label="7-дневное скользящее среднее", linewidth=2)
plt.title("Динамика средней выручки по дням")
plt.xlabel("Дата")
plt.ylabel("Средняя выручка (руб.)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n3. СРАВНЕНИЕ БУДНИ vs ВЫХОДНЫЕ")

data["Weekday"] = data["Date"].dt.weekday
data["IsWeekend"] = data["Weekday"] >= 5

weekend_stats = data.groupby("IsWeekend")[["Duration_min", "Revenue", "Avg_speed"]].mean()
weekend_stats.index = ["Будни", "Выходные"]

print("Сравнение показателей:")
print(weekend_stats)

# ИСПРАВЛЕНИЕ: Правильное построение графика для сравнения будни/выходные
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(weekend_stats.columns))
width = 0.35

plt.bar(x_pos - width/2, weekend_stats.loc["Будни"], width, label='Будни', alpha=0.8)
plt.bar(x_pos + width/2, weekend_stats.loc["Выходные"], width, label='Выходные', alpha=0.8)

plt.xlabel('Показатели')
plt.ylabel('Среднее значение')
plt.title('Сравнение показателей: будни vs выходные')
plt.xticks(x_pos, weekend_stats.columns, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n4. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ПОЕЗДОК")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data["Duration_min"], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title("Распределение длительности поездок")
plt.xlabel("Длительность (мин)")
plt.ylabel("Количество поездок")
plt.grid(True, alpha=0.3)

threshold = np.percentile(data["Duration_min"], 95)
anomalous_trips = data[data["Duration_min"] > threshold]
print(f"95-й перцентиль длительности: {threshold:.1f} мин")
print(f"Количество аномально длинных поездок: {len(anomalous_trips)}")

plt.axvline(threshold, color='red', linestyle='--', label=f'95-й перцентиль ({threshold:.1f} мин)')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(data["Avg_speed"], bins=25, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title("Распределение средних скоростей")
plt.xlabel("Средняя скорость (км/ч)")
plt.ylabel("Количество поездок")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n5. КЛАСТЕРИЗАЦИЯ АВТОМОБИЛЕЙ")

car_features = data.groupby("Car_ID").agg({
    "Revenue": "mean",
    "Duration_min": "mean",
    "Avg_speed": "mean",
    "Profit": "mean"
}).reset_index()

print("Статистика по автомобилям:")
print(car_features.describe())

features_to_scale = ["Revenue", "Duration_min", "Avg_speed"]
scaled = StandardScaler().fit_transform(car_features[features_to_scale])

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled)
car_features["Cluster"] = labels

cluster_stats = car_features.groupby("Cluster")[["Revenue", "Duration_min", "Avg_speed", "Profit"]].mean()
print("\nХарактеристики кластеров:")
print(cluster_stats)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(car_features["Revenue"], car_features["Duration_min"],
                      c=car_features["Cluster"], cmap="viridis", s=100, alpha=0.7)
plt.colorbar(scatter, label="Кластер")
plt.title("Кластеризация автомобилей")
plt.xlabel("Средняя выручка за поездку (руб.)")
plt.ylabel("Средняя длительность поездки (мин)")
plt.grid(True, alpha=0.3)

for i, cluster in enumerate(cluster_stats.index):
    plt.annotate(f'Кластер {cluster}',
                 (cluster_stats.loc[cluster, "Revenue"], cluster_stats.loc[cluster, "Duration_min"]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

print("\n6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")

correlation_features = ["Duration_min", "Distance_km", "Revenue", "Fuel_cost", "Avg_speed"]
corr = data[correlation_features].corr()

print("Корреляционная матрица:")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True,
            fmt=".2f", linewidths=0.5)
plt.title("Корреляционная матрица показателей поездок")
plt.tight_layout()
plt.show()

print("\n7. ПРОГНОЗИРОВАНИЕ ВЫРУЧКИ")

days = np.arange(len(daily_revenue))

coeffs = np.polyfit(days, daily_revenue.values, deg=3)
poly = np.poly1d(coeffs)

future_days = np.arange(len(daily_revenue) + 14)
future_pred = poly(future_days)

last_date = daily_revenue.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq='D')

plt.figure(figsize=(12, 6))
plt.plot(daily_revenue.index, daily_revenue.values, label="Фактическая средняя выручка", linewidth=2)
plt.plot(future_dates, future_pred[-14:], label="Прогноз на 14 дней", linestyle="--", linewidth=2, color='red')
plt.title("Прогноз средней выручки на 14 дней")
plt.xlabel("Дата")
plt.ylabel("Средняя выручка (руб.)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nПрогноз средней выручки на следующие 14 дней:")
for i, (date, pred) in enumerate(zip(future_dates, future_pred[-14:])):
    print(f"{date.strftime('%Y-%m-%d')}: {pred:.2f} руб.")