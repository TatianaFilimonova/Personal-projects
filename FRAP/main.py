import cv2
import numpy as np
import glob
import os

n = int(input('Введите, сколько столбцов на начале файла необходимо удалить: '))
common_mean = []

#tif_files = glob.glob('C:\Users\filim\OneDrive\Рабочий стол\FRAP для питона\Frap vez 3/*.tif')
tif_files.sort()
for filename in tif_files:
    image = cv2.imread(filename, -1)
    #    print(image) # Выводим исходный массив

    if image is None:
        print(f'Ошибка загрузки {filename}')
        continue
    print(f'Изображение {filename} успешно загружено.')

    # Удаляем краевые строки и столбцы
    image = np.delete(image, [range(n)], 0)  # строки только начало
    image = np.delete(image, [0, 1, image.shape[1] - 2, image.shape[1] - 1], 1)  # столбцы
#    print(image)

    # Используем numpy для вычисления среднего по строкам
    mean_im = np.round(np.mean(image, axis=1, dtype=np.float32))

    #    print(mean_im.tolist()) # Этот принт дает сравнить значения с тем, что получено в ориджине
    common_mean.append(mean_im.tolist())

print(common_mean)