import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def showimg_first(img, title=None):
    plt.imshow(img, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()


def showimg_second(img1, img2, title=None):
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()


def open_image(filename):
    try:
        image_gears = cv2.imread(filename, 0)
        ret, image_original = cv2.threshold(image_gears, 127, 255, cv2.THRESH_BINARY)
        return image_original
    except FileNotFoundError:
        print("Файл не найден")
        return


def erosion(image):
    circle = cv2.circle(np.zeros((100, 100), 'uint8'), (50, 50), 47, 255, -1)
    mask = cv2.circle(np.zeros((100, 100), 'uint8'), (50, 50), 50, 255, -1) - circle
    img_erosion = cv2.erode(image, mask)
    showimg_first(img_erosion, 'erosion')
    return img_erosion


def dilatation(img_erosion):
    circle = cv2.circle(np.zeros((100, 100), 'uint8'), (50, 50), 55, 255, -1)
    img_dilatation = cv2.dilate(img_erosion, circle)
    showimg_first(img_dilatation, 'dilatation')
    return img_dilatation


def original_or_dilatation(image, img_dilatation):
    img_or = cv2.bitwise_or(image, img_dilatation)
    showimg_first(img_or, 'original or dilatation')
    return img_or


def ring(img_or):
    # Размыкание (+,-) img_or элементом gear_body
    gear_body_size = 270
    gear_body = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gear_body_size, gear_body_size))
    img_gear_body = cv2.morphologyEx(img_or, cv2.MORPH_OPEN, gear_body)
    # showimg_first(img_gear_body, 'level 1')

    # Наращивание(+) gear_body элементом sampling_ring_spacer
    # Круг без шестеренок
    srs_size = 11
    sampling_ring_spacer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (srs_size, srs_size))
    img_sampling_ring_spacer = cv2.dilate(img_gear_body, sampling_ring_spacer)
    # showimg_first(img_sampling_ring_spacer, 'level 2')

    # Наращивание(+) sampling_ring_spacer элементом sampling_ring_width
    # Круг с шестеренками
    srw_size = 23
    sampling_ring_width = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (srw_size, srw_size))
    img_sampling_ring_width = cv2.dilate(img_sampling_ring_spacer, sampling_ring_width)
    # showimg_first(img_sampling_ring_spacer, 'level 3')

    # Логическое или
    # Шестеренки, в виде кольца
    img_ring = cv2.bitwise_xor(img_sampling_ring_spacer, img_sampling_ring_width)
    showimg_first(img_ring, 'ring')
    return img_ring


def original_and_ring(image, img_ring):
    img_and = cv2.bitwise_and(image, img_ring)
    showimg_first(img_and, 'original and ring')
    return img_and


def b8_dilatation_tip_spacing(img_and):
    tip_spacing_size = 25
    # дисковой элемент, где диаметр = промежутку между зубцами шестеренки
    tip_spacing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tip_spacing_size, tip_spacing_size))
    img_tip_spacing = cv2.dilate(img_and, tip_spacing)
    showimg_first(img_tip_spacing, '(original and ring) dilate tip spacing')
    return img_tip_spacing


def result(img_ring, img_tip_spacing):
    # шестеренки, в виде кольца - почти кольцо
    # B7 - B9
    img_subtract = cv2.subtract(img_ring, img_tip_spacing)
    # showimg_first(img_subtract)

    # (шестеренки, в виде кольца - почти кольцо) (+) дисковой элемент, который помогает найти изъяны зубцов
    # (B7 - B9) (+) defect_cue
    defect_cue_size = 45
    defect_cue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (defect_cue_size, defect_cue_size))
    img_dilate = cv2.dilate(img_subtract, defect_cue)
    # showimg_first(img_dilate)

    # два пятна ИЛИ почти кольцо
    # ((B7 - B9) (+) defect_cue) OR B9
    img_result = cv2.bitwise_or(img_dilate, img_tip_spacing)
    showimg_first(img_result, 'result')
    return img_result


if __name__ == '__main__':
    filename = "gears.png"
    image_original = open_image(filename)

    # B1: исходное эрозия(-) кольцом
    B1 = erosion(image_original)

    # B2: исходное дилатация(+) круг
    B2 = dilatation(B1)

    # B3: исходное ИЛИ дилатация
    B3 = original_or_dilatation(image_original, B2)

    # B7: шестеренки, в виде кольца
    B7 = ring(B3)

    # B8: исходное И кольцо
    B8 = original_and_ring(image_original, B7)

    # B9: (исходное И кольцо) дилатация(+) дисковой элемент, где диаметр = промежутку между зубцами шестеренки
    B9 = b8_dilatation_tip_spacing(B8)

    # result: ( (B7 - B9) дилатация(+)  дисковой элемент, который помогает найти изъяны зубцов) ИЛИ B9
    final_result = result(B7, B9)
