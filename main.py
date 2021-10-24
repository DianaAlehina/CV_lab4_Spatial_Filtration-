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


def showimg(img1, img2, img3, img4):
    plt.subplot(1, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('')
    plt.subplot(1, 4, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 4, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('')
    plt.subplot(1, 4, 4)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('')
    plt.show()


def open_image(filename):
    try:
        image = cv2.imread(filename, 0)
        return image
    except FileNotFoundError:
        print("Файл не найден")
        return


# нормализация (линейоное растяжение)
def normalization(image_laplacian):
    Imax = np.max(image_laplacian)
    Imin = np.min(image_laplacian)
    Omin, Omax = 0, 255
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    image = a * image_laplacian + b
    image = image.astype(np.uint8)
    return image


def image_enhancement(image_original):
    # Б: Применение оператора лапласиана к оригинальному изображению для обнаружения краев.
    image_laplacian = cv2.Laplacian(image_original, cv2.CV_64F)
    new_image_laplacian = normalization(image_laplacian)

    # В: Повышение резкости = image_original + new_image_laplacian
    image_addition = cv2.add(image_original, new_image_laplacian)

    # Г: Применение градиентного оператора Собела к оригинальному изображению
    image_sobely = cv2.convertScaleAbs(image_original)
    showimg(image_original, new_image_laplacian, image_addition, image_sobely)



if __name__ == '__main__':
    filename = "skeleton.jpg"
    image_original = open_image(filename)
    # showimg_first(image_original)
    image_enhancement(image_original)
