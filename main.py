import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def showimg_second(img1, img2, title1=None, title2=None):
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    if title1 is not None:
        plt.title(title1)
    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    if title2 is not None:
        plt.title(title2)
    plt.show()


def showimg(img1, img2, img3, img4, img5, img6, img7, img8):
    plt.subplot(2, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2, 4, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Laplacian')
    plt.subplot(2, 4, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Original + Laplacian')
    plt.subplot(2, 4, 4)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Sobel')
    plt.subplot(2, 4, 5)
    plt.imshow(img5, cmap='gray')
    plt.axis('off')
    plt.title('Smoothing, 5x5')
    plt.subplot(2, 4, 6)
    plt.imshow(img6, cmap='gray')
    plt.axis('off')
    plt.title('Mask=(Original+Laplacian)*Smoothing')
    plt.subplot(2, 4, 7)
    plt.imshow(img7, cmap='gray')
    plt.axis('off')
    plt.title('Original + Mask')
    plt.subplot(2, 4, 8)
    plt.imshow(img8, cmap='gray')
    plt.axis('off')
    plt.title('Gamma correction')
    plt.show()


def open_image(filename):
    try:
        image = cv2.imread(filename)
        return image
    except FileNotFoundError:
        print("Файл не найден")
        return


# нормализация(линейоное растяжение)
def normalization(image):
    Imax = np.max(image)
    Imin = np.min(image)
    Omin, Omax = 0, 255
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    image = a * image + b
    image = image.astype(np.uint8)
    return image


def image_enhancement(image_original):
    image_original = np.uint8(image_original)

    # Б: Применение оператора лапласиана к оригинальному изображению для обнаружения краев.
    image_laplacian = cv2.Laplacian(image_original, cv2.CV_64F)
    # Используем нормализацию, чтобы картинка не выходила за отрезок [0, 255]
    image_laplacian = normalization(image_laplacian)
    image_laplacian = np.uint8(image_laplacian)

    # В: Повышение резкости = image_original + image_laplacian
    image_addition = cv2.addWeighted(image_original, 1, image_laplacian, 1, 0)
    image_addition = np.uint8(image_addition)
    image_addition = normalization(image_addition)

    # Г: Применение градиентного оператора Собела к оригинальному изображению
    image_sobelx = cv2.Sobel(image_original, cv2.CV_64F, 1, 0, 3)
    image_sobely = cv2.Sobel(image_original, cv2.CV_64F, 0, 1, 3)
    image_sobelx = cv2.convertScaleAbs(image_sobelx)
    image_sobely = cv2.convertScaleAbs(image_sobely)
    image_sobelxy = cv2.add(image_sobelx, image_sobely)
    image_sobelxy = np.uint8(image_sobelxy)

    # Д: Сглаживание градиентного изображения image_sobelxy, сохранив края изображения, size = 5
    image_medianBlur = cv2.medianBlur(image_sobelxy, 5)
    image_medianBlur = np.uint8(image_medianBlur)

    # E: Изображение-маска = image_addition * image_medianBlur
    image_mask = 255 * (image_addition / 255 * image_medianBlur / 255)
    image_mask = np.uint8(image_mask)

    # Ж: Повышение резкости = image_original + image_mask
    image_addition2 = cv2.add(image_original, image_mask)
    image_addition2 = np.uint8(image_addition2)

    # З: Градационная коррекция по степенному закону
    gamma = 0.5
    image_correction = 255 * (image_addition2 / 255) ** gamma
    image_correction = np.uint8(image_correction)

    # Показ изображений
    showimg_second(image_original, image_laplacian, 'Original', 'Laplacian')
    showimg_second(image_addition, image_sobelxy, 'Original + Laplacian', 'Sobel')
    showimg_second(image_medianBlur, image_mask, 'Smoothing, 5x5', 'Mask=(Original+Laplacian)*Smoothing')
    showimg_second(image_addition2, image_correction, 'Original + Mask', 'Gamma correction')
    showimg(image_original, image_laplacian, image_addition, image_sobelxy, image_medianBlur, image_mask,
            image_addition2, image_correction)


if __name__ == '__main__':
    filename = "skeleton.jpg"
    image_original = open_image(filename)
    image_enhancement(image_original)
