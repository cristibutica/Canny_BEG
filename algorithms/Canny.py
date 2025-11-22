import cv2 as cv
import numpy as np

# --- MODIFIED Canny Function ---
max_lowThreshold = 100

title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3

def my_Canny(image, low_threshold, high_threshold, kernel_size=3):
    """
    Implementare Canny Edge Detection optimizată (Vectorizată).
    Folosește operații NumPy și ConnectedComponents pentru a evita buclele 'for'.
    """
    # 1. Reducerea Zgomotului
    img_blur = cv.GaussianBlur(image, (kernel_size, kernel_size), 1.4)

    # 2. Calculul Gradientului
    # Folosim float64 pentru precizie la calcule intermediare
    gx = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=3)

    # Magnitudine și Unghi
    magnitude = cv.magnitude(gx, gy)  # Funcție optimizată din OpenCV
    angle = cv.phase(gx, gy, angleInDegrees=True)  # Returnează 0 - 360

    # 3. Suprimarea Non-Maximelor (NMS) - FĂRĂ BUCLE FOR
    # Cuantizăm unghiurile în 4 direcții: 0, 45, 90, 135
    # Facem o copie cu border pentru a putea "decala" imaginea ușor
    M, N = image.shape
    # Unghiurile sunt simplificate la 0, 45, 90, 135 (simetric față de 180)
    angle = angle % 180

    # Creăm măști pentru direcții
    mask_0 = (angle < 22.5) | (angle >= 157.5)
    mask_45 = (angle >= 22.5) & (angle < 67.5)
    mask_90 = (angle >= 67.5) & (angle < 112.5)
    mask_135 = (angle >= 112.5) & (angle < 157.5)

    # Imaginea "supresată" inițializată cu 0
    Z = np.zeros_like(magnitude)

    # --- VECTORIZARE PENTRU VECINI ---
    # Shiftăm matricele pentru a compara cu vecinii fără bucle
    # Padăm magnitudinea cu zerouri pe margini pentru a evita erorile de indexare la shift
    mag_pad = np.pad(magnitude, ((1, 1), (1, 1)), mode='constant')

    # Vecinii: (x, y) corespund la mag_pad[1:-1, 1:-1]
    # Est-Vest (0 grade): (1, 0) și (1, 2) în coordonate de pad
    mag_w = mag_pad[1:-1, :-2]  # West
    mag_e = mag_pad[1:-1, 2:]  # East

    # Nord-Sud (90 grade)
    mag_n = mag_pad[:-2, 1:-1]  # North
    mag_s = mag_pad[2:, 1:-1]  # South

    # Diagonale
    mag_ne = mag_pad[:-2, 2:]  # North-East
    mag_sw = mag_pad[2:, :-2]  # South-West
    mag_nw = mag_pad[:-2, :-2]  # North-West
    mag_se = mag_pad[2:, 2:]  # South-East

    # Aplicăm NMS folosind măștile de unghiuri
    # Păstrăm pixelul doar dacă e mai mare decât vecinii săi pe direcția gradientului
    keep_0 = mask_0 & (magnitude >= mag_w) & (magnitude >= mag_e)
    keep_90 = mask_90 & (magnitude >= mag_n) & (magnitude >= mag_s)
    keep_45 = mask_45 & (magnitude >= mag_nw) & (magnitude >= mag_se)  # Atentie la coordonate vs unghi visual
    keep_135 = mask_135 & (magnitude >= mag_ne) & (magnitude >= mag_sw)

    # Combinăm toate muchiile păstrate
    nms_mask = keep_0 | keep_90 | keep_45 | keep_135
    Z[nms_mask] = magnitude[nms_mask]

    # 4. Hysteresis Thresholding (Optimizat)
    # Identificăm muchiile sigure (Strong) și posibile (Weak)
    strong_mask = Z >= high_threshold

    # Dacă Low este 0, totul devine muchie, ceea ce arată urât. Punem un minim logic.
    safe_low = max(low_threshold, 1)
    potential_mask = Z >= safe_low

    # Folosim Connected Components pentru a găsi muchiile weak conectate de strong
    # Această funcție este foarte rapidă în OpenCV
    # Etichetăm toate zonele "potențiale"
    num_labels, labels = cv.connectedComponents(potential_mask.astype(np.uint8))

    # Găsim care etichete (componente) conțin cel puțin un pixel "strong"
    # labels * strong_mask va lăsa doar label-urile din zonele strong.
    # np.unique ne dă lista ID-urilor de componente valide.
    valid_labels = np.unique(labels[strong_mask])

    # Ignorăm label-ul 0 (background) dacă a fost prins
    valid_labels = valid_labels[valid_labels > 0]

    # Creăm masca finală: pixelii care fac parte dintr-o componentă validă
    # np.isin este relativ rapid
    final_mask = np.isin(labels, valid_labels)

    return (final_mask * 255).astype(np.uint8)

def CannyThreshold(low_threshold, high_threshold, src, src_gray):
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = my_Canny(img_blur, low_threshold, high_threshold, kernel_size)
    mask = detected_edges != 0
    # Create the 3-channel BGR image
    dst = src * (mask[:, :, None].astype(src.dtype))

    # Return the final image
    return dst

