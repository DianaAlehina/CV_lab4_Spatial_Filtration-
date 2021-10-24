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


def showimg(img1, img2, img3, img4, title=None):
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
    if title is not None:
        plt.title(title)
    plt.show()


def open_image(filename):
    try:
        image = cv2.imread(filename, 0)
        return image
    except FileNotFoundError:
        print("Файл не найден")
        return


# нормализация(линейоное растяжение)
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
    image_addition = cv2.addWeighted(image_original, 1, new_image_laplacian, 1, 0.0)

    # Г: Применение градиентного оператора Собела к оригинальному изображению
    image_sobelx = cv2.Sobel(image_original, cv2.CV_64F, 1, 0, ksize=3)
    image_sobely = cv2.Sobel(image_original, cv2.CV_64F, 0, 1, ksize=3)
    image_sobelx = cv2.convertScaleAbs(image_sobelx)
    image_sobely = cv2.convertScaleAbs(image_sobely)
    image_sobelxy = cv2.addWeighted(image_sobelx, 1, image_sobely, 1, 0)

    # Д:
    img_mask = cv2.blur(image_sobelxy, (5, 5))

    # E:
    img_mask1 = cv2.bitwise_and(image_addition, img_mask)

    # Ж:
    img_mask2 = cv2.addWeighted(image_original, 1, img_mask1, 0.4, 1.0) #cv2.add(image_original, img_mask1)
    # З:
    gamma = 0.5
    img_mask3 = np.array(255 * (img_mask2 / 255) ** gamma, dtype='uint8')

    showimg(image_original, new_image_laplacian, image_addition, image_sobelxy)
    showimg(img_mask, img_mask1, img_mask2, img_mask3)


if __name__ == '__main__':
    filename = "skeleton.jpg"
    image_original = open_image(filename)
    # showimg_first(image_original)
    image_enhancement(image_original)
