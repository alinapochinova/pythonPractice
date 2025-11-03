import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


def create_weather_csv(filename="weather.csv", days=15):
    """
    Автоматически создает CSV-файл с погодными данными
    """
    if not os.path.isabs(filename):
        filename = os.path.join(os.getcwd(), filename)

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    date_strings = [date.strftime("%Y-%m-%d") for date in dates]

    np.random.seed(42)

    temperature = np.random.randint(0, 15, days)
    humidity = np.random.randint(60, 90, days)

    weather_data = {
        "Date": date_strings,
        "Temperature": temperature,
        "Humidity": humidity
    }

    df = pd.DataFrame(weather_data)
    df.to_csv(filename, index=False)
    print(f"Файл создан: {filename}")
    print(f"Сгенерировано {days} дней данных")
    print(f"Текущая директория: {os.getcwd()}")

    return df, filename


class WeatherAnalyzer:
    def __init__(self, file_path, analysis_days=5):
        """
        Инициализация объекта и загрузка данных из CSV-файла
        """
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        self.file_path = file_path
        self.analysis_days = analysis_days

        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден. Создаем тестовые данные...")
            _, self.file_path = create_weather_csv(file_path)

        self.full_data = pd.read_csv(self.file_path)

        self.data = self.full_data.head(analysis_days)

        print(f"Погодные данные успешно загружены из: {self.file_path}")
        print(f"Загружено {len(self.full_data)} записей")
        print(f"Для анализа используется {len(self.data)} дней")

    def show_head(self, n=5):
        """
        Показывает первые n строк данных
        """
        print(f"Первые {n} строк данных (для анализа):")
        print(self.data.head(n))

    def get_average_temperature(self):
        """
        Вычисляет среднюю температуру за период анализа
        """
        avg_temp = self.data["Temperature"].mean()
        print(f"Средняя температура за {len(self.data)} дней анализа: {avg_temp:.1f}°C")
        return avg_temp

    def get_extreme_days(self):
        """
        Находит дни с максимальной и минимальной температурой в периоде анализа
        """
        max_temp_idx = self.data["Temperature"].idxmax()
        min_temp_idx = self.data["Temperature"].idxmin()

        max_temp_day = self.data.loc[max_temp_idx]
        min_temp_day = self.data.loc[min_temp_idx]

        extreme_days = pd.DataFrame({
            "Тип": ["Максимальная", "Минимальная"],
            "Дата": [max_temp_day["Date"], min_temp_day["Date"]],
            "Температура": [max_temp_day["Temperature"], min_temp_day["Temperature"]],
            "Влажность": [max_temp_day["Humidity"], min_temp_day["Humidity"]]
        })

        print(f"Дни с экстремальными температурами (из {len(self.data)} дней анализа):")
        print(extreme_days)

        return extreme_days

    def plot_temperature_trend(self):
        """
        Строит график изменения температуры по дням (только для дней анализа)
        """
        plt.figure(figsize=(10, 5))

        plt.plot(range(len(self.data)), self.data["Temperature"],
                 marker='o', linewidth=2, markersize=8, color='red')

        plt.title(f"Изменение температуры за {len(self.data)} дней")
        plt.xlabel("Дни")
        plt.ylabel("Температура (°C)")
        plt.grid(True, alpha=0.3)

        plt.xticks(range(len(self.data)), self.data["Date"], rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_temperature_humidity(self):
        """
        Строит scatter plot зависимости температуры от влажности (только для дней анализа)
        """
        plt.figure(figsize=(8, 6))

        plt.scatter(self.data["Temperature"], self.data["Humidity"],
                    alpha=0.7, s=100, color='blue')

        plt.title(f"Зависимость температуры от влажности ({len(self.data)} дней)")
        plt.xlabel("Температура (°C)")
        plt.ylabel("Влажность (%)")
        plt.grid(True, alpha=0.3)

        for i, (temp, hum) in enumerate(zip(self.data["Temperature"], self.data["Humidity"])):
            plt.annotate(self.data["Date"][i], (temp, hum),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10, alpha=0.8)

        plt.tight_layout()
        plt.show()

    def save_summary(self, output_path="weather_summary.csv"):
        """
        Сохраняет результаты анализа в новый CSV-файл
        """

        if not os.path.isabs(output_path):
            output_path = os.path.join(os.getcwd(), output_path)

        summary_data = []

        avg_temp = self.data["Temperature"].mean()
        max_temp = self.data["Temperature"].max()
        min_temp = self.data["Temperature"].min()
        avg_humidity = self.data["Humidity"].mean()

        summary_data.extend([
            {"Категория": "Общая статистика", "Параметр": "Количество дней анализа", "Значение": len(self.data)},
            {"Категория": "Общая статистика", "Параметр": "Средняя температура", "Значение": f"{avg_temp:.1f}°C"},
            {"Категория": "Общая статистика", "Параметр": "Максимальная температура", "Значение": f"{max_temp:.1f}°C"},
            {"Категория": "Общая статистика", "Параметр": "Минимальная температура", "Значение": f"{min_temp:.1f}°C"},
            {"Категория": "Общая статистика", "Параметр": "Средняя влажность", "Значение": f"{avg_humidity:.1f}%"}
        ])

        max_temp_idx = self.data["Temperature"].idxmax()
        min_temp_idx = self.data["Temperature"].idxmin()

        max_temp_day = self.data.loc[max_temp_idx]
        min_temp_day = self.data.loc[min_temp_idx]

        summary_data.extend([
            {"Категория": "Экстремальные дни", "Параметр": "Самый теплый день",
             "Значение": f"{max_temp_day['Date']} ({max_temp_day['Temperature']}°C)"},
            {"Категория": "Экстремальные дни", "Параметр": "Самый холодный день",
             "Значение": f"{min_temp_day['Date']} ({min_temp_day['Temperature']}°C)"}
        ])

        correlation = np.corrcoef(self.data["Temperature"], self.data["Humidity"])[0, 1]
        summary_data.append({
            "Категория": "Статистика",
            "Параметр": "Корреляция температура-влажность",
            "Значение": f"{correlation:.2f}"
        })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"Результаты анализа сохранены в файл: {output_path}")
        print(f"Полный путь: {os.path.abspath(output_path)}")

        print(f"\nСодержимое сохраненного файла:")
        print(summary_df)

if __name__ == "__main__":
    analyzer = WeatherAnalyzer("weather.csv")
    print()

    analyzer.show_head()
    print()

    analyzer.get_average_temperature()
    print()

    analyzer.get_extreme_days()
    print()

    print("Строим график изменения температуры...")
    analyzer.plot_temperature_trend()

    print("Строим scatter plot зависимости температуры от влажности...")
    analyzer.plot_temperature_humidity()

    print("Сохраняем результаты анализа...")
    analyzer.save_summary()
