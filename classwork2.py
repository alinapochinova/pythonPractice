import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

weekdays = np.random.normal(200, 40, 4)
friday = np.random.normal(150, 30, 1)
weekend = np.random.normal(300, 50, 2)

visitors = np.concatenate([weekdays, friday, weekend])

print(f"Общее число посетителей за неделю: {visitors.sum():.0f}")
print(f"Среднее посещений в день: {np.mean(visitors):.1f}")
print(f"Медиана посещений в день: {np.median(visitors):.1f}")

days_of_week = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
max_day_index = np.argmax(visitors)
print(f"Максимум посетителей был в {days_of_week[max_day_index]} — {visitors[max_day_index]:.0f} человек")

plt.figure(figsize=(10, 6))

colors = ['blue', 'blue', 'blue', 'blue', 'blue', 'orange', 'orange']

bars = plt.bar(days_of_week, visitors, color=colors, alpha=0.7)
plt.title("Количество посетителей фитнес-клуба по дням недели")
plt.xlabel("День недели")
plt.ylabel("Количество посетителей")

for i, v in enumerate(visitors):
    plt.text(i, v + 5, f'{v:.0f}', ha='center', va='bottom')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label='Будни'),
    Patch(facecolor='orange', alpha=0.7, label='Выходные')
]
plt.legend(handles=legend_elements)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
