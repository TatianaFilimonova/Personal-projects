import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.signal import find_peaks
import cv2
import statistics
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import shutil  # Для копирования файлов
from pathlib import Path  # Для работы с путями

# Функция Гаусса в заданной форме
def gaussian(x, y0, A, xc, w):
    return y0 + (A / (w * np.sqrt(np.pi / (4 * np.log(2))))) * np.exp(-4 * np.log(2) * (x - xc)**2 / w**2)

# Функция для обработки изображения
def process_image(image_path, reference_image_path=None, top=0, bottom=0, left=0, right=0):
    try:
        # Преобразуем пути в объекты Path
        image_path = Path(image_path)
        reference_image_path = Path(reference_image_path) if reference_image_path else None

        # Временная директория для копирования файлов
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        # Копируем файлы во временную директорию с ASCII-именами
        temp_image_path = temp_dir / "image.tif"
        shutil.copy(str(image_path), str(temp_image_path))

        if reference_image_path:
            temp_reference_path = temp_dir / "reference.tif"
            shutil.copy(str(reference_image_path), str(temp_reference_path))
        else:
            temp_reference_path = None

        # Загрузка изображений с учетом 16-битной глубины
        im00 = cv2.imread(str(temp_reference_path), cv2.IMREAD_ANYDEPTH) if temp_reference_path else None
        im02 = cv2.imread(str(temp_image_path), cv2.IMREAD_ANYDEPTH)

        # Проверка, что изображения загружены
        if im00 is None or im02 is None:
            raise ValueError("Ошибка загрузки изображения. Проверьте путь и формат файла.")

        # Удаление строк и столбцов для эталонного изображения
        if im00 is not None:
            im00 = np.delete(im00, range(top), 0)  # Удаление строк сверху
            im00 = np.delete(im00, range(len(im00) - bottom, len(im00)), 0)  # Удаление строк снизу
            im00 = np.delete(im00, range(left), 1)  # Удаление столбцов слева
            im00 = np.delete(im00, range(len(im00[0]) - right, len(im00[0])), 1)  # Удаление столбцов справа
            im00 = im00.astype(np.float64) / np.max(im00)  # Нормализация
            mean_before = [statistics.mean(im00[i]) if statistics.mean(im00[i]) != 0 else 1e-10 for i in range(len(im00))]
        else:
            mean_before = [1] * (1024 - top - bottom)

        # Удаление строк и столбцов для основного изображения
        im02 = np.delete(im02, range(top), 0)  # Удаление строк сверху
        im02 = np.delete(im02, range(len(im02) - bottom, len(im02)), 0)  # Удаление строк снизу
        im02 = np.delete(im02, range(left), 1)  # Удаление столбцов слева
        im02 = np.delete(im02, range(len(im02[0]) - right, len(im02[0])), 1)  # Удаление столбцов справа
        im02 = im02.astype(np.float64) / np.max(im02)  # Нормализация
        mean_02 = [statistics.mean(im02[i]) for i in range(len(im02))]

        # Нормализация данных
        norm_03 = [mean_02[i] / mean_before[i] for i in range(len(mean_before))]
        data = norm_03

        # Значения по X
        x_values = np.linspace(1, 425, len(data))

        # Проверка на NaN и inf
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Данные содержат NaN или inf. Проверьте входные данные.")

        # Поиск локальных минимумов (используем инвертированные данные)
        inverted_data = -np.array(data)
        peaks, _ = find_peaks(inverted_data, height=np.mean(inverted_data))

        # Определить направление колокола
        if len(peaks) == 0:
            peak_value = np.max(data)
            peak_index = np.argmax(data)
            A_initial = (peak_value - np.mean(data)) * 10
        else:
            min_index = peaks[np.argmin(np.array(data)[peaks])]
            peak_value = data[min_index]
            peak_index = min_index
            A_initial = (peak_value - np.mean(data)) * 10

        # Начальные параметры для аппроксимации
        p0 = [
            np.mean(data),
            A_initial,
            x_values[peak_index],
            10,
        ]

        # Аппроксимация данных функцией Гаусса
        params, cov = curve_fit(gaussian, x_values, data, p0=p0, maxfev=10000)

        # Создание гладкой сетки для построения аппроксимирующей кривой
        x_smooth = np.linspace(1, 425, 1000)
        y_smooth = gaussian(x_smooth, *params)

        # Вычисление полуширины (FWHM)
        y0, A, xc, w = params
        y_max = gaussian(xc, *params)
        y_half = y0 + (y_max - y0) / 2

        # Решение уравнения для нахождения точек полуширины
        def equation(x):
            return gaussian(x, *params) - y_half

        x_left_guess = xc - w
        x_right_guess = xc + w

        x_left = fsolve(equation, x_left_guess)[0]
        x_right = fsolve(equation, x_right_guess)[0]

        fwhm = x_right - x_left

        # Очистка предыдущего графика
        for widget in graph_frame.winfo_children():
            widget.destroy()

        # Создание нового графика
        fig = Figure(figsize=(6, 4), dpi=100)
        plot = fig.add_subplot(111)
        plot.plot(x_values, data, 'b-', label='Исходные данные')
        plot.plot(x_smooth, y_smooth, 'r-', label='Аппроксимация (Гаусс)')
        plot.axhline(y_half, color='gray', linestyle='--', label='Половина высоты')
        plot.axvline(x_left, color='green', linestyle='--', label=f'x_left = {x_left:.2f}')
        plot.axvline(x_right, color='purple', linestyle='--', label=f'x_right = {x_right:.2f}')
        plot.set_xlabel('X')
        plot.set_ylabel('Значение')
        plot.legend()
        plot.set_title(f'Аппроксимация данных функцией Гаусса\nПолуширина (FWHM) = {fwhm:.2f}')

        # Встраивание графика в интерфейс
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Вывод параметров аппроксимации
        print(f"Смещение (y0): {params[0]}")
        print(f"Площадь (A): {params[1]}")
        print(f"Центр (xc): {params[2]}")
        print(f"Полуширина (w): {params[3]}")
        print(f"Полуширина (FWHM): {fwhm:.2f}")

        # Удаление временных файлов
        temp_image_path.unlink()
        if temp_reference_path:
            temp_reference_path.unlink()

    except Exception as e:
        print(f"Ошибка: {e}")

# Функции для загрузки изображений
def load_reference_image():
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
    if file_path:
        reference_image_path.set(file_path)
        file_name = os.path.basename(file_path)  # Извлекаем имя файла
        reference_status_label.config(text=f'Эталонное изображение "{file_name}" загружено', fg="green")
        print(f"Эталонное изображение загружено: {file_path}")

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
    if file_path:
        image_path.set(file_path)
        file_name = os.path.basename(file_path)  # Извлекаем имя файла
        image_status_label.config(text=f'Изображение "{file_name}" загружено', fg="green")
        print(f"Изображение загружено: {file_path}")

# Создание GUI
root = tk.Tk()
root.title("Аппроксимация данных функцией Гаусса")

reference_image_path = tk.StringVar()
image_path = tk.StringVar()

# Метки для отображения статуса загрузки
reference_status_label = tk.Label(root, text="Эталонное изображение не загружено", fg="red")
reference_status_label.grid(row=0, column=1, padx=5, pady=5)

image_status_label = tk.Label(root, text="Изображение не загружено", fg="red")
image_status_label.grid(row=1, column=1, padx=5, pady=5)

# Кнопки для загрузки изображений
tk.Button(root, text="Загрузить эталонное изображение", command=load_reference_image).grid(row=0, column=0, padx=5, pady=5)
tk.Button(root, text="Загрузить изображение", command=load_image).grid(row=1, column=0, padx=5, pady=5)

# Поля для ввода количества строк и столбцов для удаления
tk.Label(root, text="Строк сверху:").grid(row=2, column=0, padx=5, pady=5)
top_entry = tk.Entry(root)
top_entry.grid(row=2, column=1, padx=5, pady=5)
top_entry.insert(0, "0")  # Значение по умолчанию

tk.Label(root, text="Строк снизу:").grid(row=3, column=0, padx=5, pady=5)
bottom_entry = tk.Entry(root)
bottom_entry.grid(row=3, column=1, padx=5, pady=5)
bottom_entry.insert(0, "0")  # Значение по умолчанию

tk.Label(root, text="Столбцов слева:").grid(row=4, column=0, padx=5, pady=5)
left_entry = tk.Entry(root)
left_entry.grid(row=4, column=1, padx=5, pady=5)
left_entry.insert(0, "0")  # Значение по умолчанию

tk.Label(root, text="Столбцов справа:").grid(row=5, column=0, padx=5, pady=5)
right_entry = tk.Entry(root)
right_entry.grid(row=5, column=1, padx=5, pady=5)
right_entry.insert(0, "0")  # Значение по умолчанию

# Кнопка для обработки изображений
tk.Button(root, text="Обработать", command=lambda: process_image(
    image_path.get(),
    reference_image_path.get(),
    int(top_entry.get()),
    int(bottom_entry.get()),
    int(left_entry.get()),
    int(right_entry.get()))
).grid(row=6, column=0, columnspan=2, padx=5, pady=5)

# Фрейм для графика
graph_frame = tk.Frame(root)
graph_frame.grid(row=7, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()