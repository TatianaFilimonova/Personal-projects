import math
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
print(f"Выбран файл: {file_path}")

def text_to_array(filename): #открываем файл и переводим данные в удобный вид
    array = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts

                # На всякий случай заменяем запятые на точки
                value = value.replace(',', '.')
                # Определяем тип данных, тоже на всякий случай
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

                array.append([key, value])

    return array

sorted_array = sorted(text_to_array(file_path), key=lambda x: x[0]) # Сортирую имена переменных по алфавиту

print("Отсортированный список переменных:\n")
for i, (key, value) in enumerate(sorted_array, 1):
    print(f"{i-1}. {key}: {value} ({type(value).__name__})")

def DupuiEquation(varibles, time):
    Q = []
    for i in range(1, time):
        q = (varibles[5][1] * varibles[10][1] * (varibles[3][1] - varibles[1][1])) \
            / (18.41 * varibles[0][1] * varibles[2][1] * math.log((varibles[5][1] * i) / (varibles[4][1] * varibles[0][1] * varibles[8][1] * (varibles[7][1])**2 )))
        Q.append(q)

    return Q

t = int(input('\n\nВведите горизонт прогноза в сутках: '))
y = DupuiEquation(sorted_array, t)


plt.plot(np.linspace(1, t, len(y)), y, color='red')
plt.title('Дебит скважины (по формуле Дюпюи) для неустановившегося режима', wrap=True)
plt.xlabel('Время, сут')
plt.ylabel('Дебит нефти, $м^3/сут$')
plt.grid()

plt.savefig('Graph.png') # Дефолтное сохранение в ту же папку

# Неработающее сохранение графика через
'''root = tk.Tk()
root.withdraw()
file_path = filedialog.asksaveasfilename(
    title="Сохранить график",
    defaultextension=".png",
    filetypes=[
        ("PNG files", "*.png"),
        ("JPEG files", "*.jpg"),
        ("PDF files", "*.pdf"),
        ("SVG files", "*.svg"),
        ("All files", "*.*")
    ]
)
if file_path:
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {file_path}")
else:
    print("Сохранение отменено")
    
root.destroy()'''

plt.show()