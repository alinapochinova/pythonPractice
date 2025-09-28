import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

informatics = np.clip(np.random.normal(3.5, 1.5, 250), 0, None)
economics = np.clip(np.random.normal(2.5, 1.0, 250), 0, None)
philology = np.clip(np.random.normal(2.0, 0.8, 250), 0, None)

plt.figure(figsize=(10, 6))
plt.hist(informatics, bins=20, alpha=0.6, label="Информатика")
plt.hist(economics, bins=20, alpha=0.6, label="Экономика")
plt.hist(philology, bins=20, alpha=0.6, label="Филология")
plt.title("Распределение времени в соцсетях по факультетам")
plt.xlabel("Время, часы")
plt.ylabel("Количество студентов")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot([informatics, economics, philology], labels=["Информатика", "Экономика", "Филология"])
plt.title("Сравнение времени в соцсетях (boxplot)")
plt.ylabel("Время, часы")
plt.show()

for faculty, data in zip(["Информатика", "Экономика", "Филология"], [informatics, economics, philology]):
    print(f"{faculty}: Среднее={np.mean(data):.2f}, "
          f"Медиана={np.median(data):.2f}, "
          f"Std={np.std(data):.2f}")

print("\nПроцент студентов, проводящих в соцсетях более 4 часов в день:")
for faculty, data in zip(["Информатика", "Экономика", "Филология"], [informatics, economics, philology]):
    above_4h = np.sum(data > 4.0)
    percentage = (above_4h / len(data)) * 100
    print(f"{faculty}: {above_4h}/{len(data)} студентов ({percentage:.1f}%)")
