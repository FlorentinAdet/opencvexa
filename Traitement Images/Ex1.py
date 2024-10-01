import cv2
import numpy as np
import os

# Créer un dossier 'images' s'il n'existe pas déjà
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Charger l'image en niveaux de gris
image_path = os.getcwd()+'/Traitement Images/images/landscape.jpg'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("L'image n'a pas pu être chargée. Vérifiez le chemin spécifié.")
    exit()

# Filtre Sobel (Contours verticaux et horizontaux)
def sobel_filter(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # Sobel horizontal
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Sobel vertical
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel)
    return sobel

# Transformation de Fourier
def fourier_transform(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

    return dft_shift, magnitude_spectrum

# Transformation inverse de Fourier
def inverse_fourier_transform(dft_shift):
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    
    return img_back

# Segmentation par seuillage adaptatif
def adaptive_threshold(img):
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY, 11, 2)
    return th

# Fonction pour sauvegarder l'image
def save_image(image, filename):
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

# Afficher les fenêtres et quitter avec les touches spécifiques
def main():
    sobel = sobel_filter(img)
    dft_shift, magnitude_spectrum = fourier_transform(img)
    img_back = inverse_fourier_transform(dft_shift)
    thresholded = adaptive_threshold(img)
    closed_windows = 0

    # Sauvegarder les images après transformation
    save_image(sobel, os.getcwd()+'/Traitement Images/images/sobel_contours.png')
    save_image(np.uint8(magnitude_spectrum), os.getcwd()+'/Traitement Images/images/fourier_spectrum.png')
    save_image(img_back, os.getcwd()+'/Traitement Images/images/inverse_fourier.png')
    save_image(thresholded, os.getcwd()+'/Traitement Images/images/adaptive_threshold.png')

    # Afficher les fenêtres
    cv2.imshow('Sobel - Détection de contours', sobel)
    cv2.imshow('Transformation de Fourier - Spectre de fréquence', magnitude_spectrum)
    cv2.imshow('Transformation inverse de Fourier', img_back)
    cv2.imshow('Segmentation par seuillage adaptatif', thresholded)

    # Boucle pour fermer chaque fenêtre une par une
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            cv2.destroyWindow('Sobel - Détection de contours')
            closed_windows += 1
        elif key == ord('2'):
            cv2.destroyWindow('Transformation de Fourier - Spectre de fréquence')
            closed_windows += 1
        elif key == ord('3'):
            cv2.destroyWindow('Transformation inverse de Fourier')
            closed_windows += 1
        elif key == ord('4'):
            cv2.destroyWindow('Segmentation par seuillage adaptatif')
            closed_windows += 1
        
        # Quitter automatiquement une fois que toutes les fenêtres sont fermées
        if closed_windows == 4:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
