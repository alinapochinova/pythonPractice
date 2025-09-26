import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

moscow = np.random.normal(120_000, 25_000, 300)
saint_p = np.random.normal(90_000, 20_000, 300)
kazan = np.random.normal(70_000, 15_000, 300)

plt.figure(figsize=(10, 6))
plt.hist(moscow, bins=20, alpha=0.6, label="Москва")
plt.hist(saint_p, bins=20, alpha=0.6, label="Санкт-Петербург")
plt.hist(kazan, bins=20, alpha=0.6, label="Казань")
plt.title("Распределение расходов семей")
plt.xlabel("Расходы, руб.")
plt.ylabel("Количество семей")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([moscow, saint_p, kazan], labels=["Москва", "Санкт-Петербург", "Казань"])
plt.title("Сравнение расходов семей (boxplot)")
plt.ylabel("Расходы, руб.")
plt.show()

for city, data in zip(["Москва", "Санкт-Петербург", "Казань"], [moscow, saint_p, kazan]):
    print(f"{city}: Среднее={np.mean(data):.0f}, "
          f"Медиана={np.median(data):.0f}, "
          f"Std={np.std(data):.0f}")

print("\nПроцент семей с расходами выше 100000 рублей:")
for city, data in zip(["Москва", "Санкт-Петербург", "Казань"], [moscow, saint_p, kazan]):
    above_100k = np.sum(data > 100_000)
    percentage = (above_100k / len(data)) * 100
    print(f"{city}: {above_100k}/{len(data)} семей ({percentage:.1f}%)")