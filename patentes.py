import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure(figsize=(10, 10))
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

# Lista de imágenes de autos
image_files = ['./img/img01.png', './img/img02.png', './img/img03.png', './img/img04.png', './img/img05.png', 
               './img/img06.png', './img/img07.png', './img/img08.png', './img/img09.png', './img/img10.png', 
               './img/img11.png', './img/img12.png']

# Procesar cada imagen de auto
for image_file in image_files:
    # Leer la imagen en escala de grises y en RGB
    img_auto_gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    img_auto = cv2.imread(image_file)
    img_auto_rgb = cv2.cvtColor(img_auto, cv2.COLOR_BGR2RGB)

    # Aplicar umbral binario
    _, img_binary = cv2.threshold(img_auto_gray, 120, 255, cv2.THRESH_BINARY)

    # Aplicar la operación de top hat para resaltar detalles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_tophat = cv2.morphologyEx(img_auto_gray, kernel=kernel, op=cv2.MORPH_TOPHAT)

    # Binarización de la imagen top hat
    _, img_binary_tophat = cv2.threshold(img_tophat, 55, 255, cv2.THRESH_BINARY)

    # Intersección entre la imagen binaria y la imagen de top hat
    intersection = np.bitwise_and(img_binary, img_binary_tophat)

    # Encontrar contornos en la imagen resultante
    contours, hierarchy = cv2.findContours(intersection, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Filtrar los contornos basados en su tamaño y proporción
    filtered_contours = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        ratio = h / w
        area = w * h
        if 0.6 < ratio  and 45 < area < 250 and 10 <= h <= 20:
            filtered_contours.append((x, y, w, h, contours[i]))

    # Agrupar contornos cercanos
    groups = []
    for i, (x1, y1, w1, h1, contour1) in enumerate(filtered_contours):
        found_group = False
        for group in groups:
            for (x2, y2, w2, h2, contour2) in group:
                if abs(x1 - x2) <= 80 and abs(y1 - y2) <= 5:
                    group.append((x1, y1, w1, h1, contour1))
                    found_group = True
                    break
            if found_group:
                break
        if not found_group:
            groups.append([(x1, y1, w1, h1, contour1)])

    # Encontrar el grupo más grande
    if groups:
        largest_group = max(groups, key=len)
        min_x, min_y = 10000, 10000
        max_x, max_y = 0, 0
        max_w = 0
        # Encontrar las coordenadas de la patente
        for (x, y, w, h, contour) in largest_group:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_w = max(max_w, w)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        # Asegurarse de que las coordenadas están dentro de los límites
        h_img, w_img = intersection.shape[:2]
        min_y = np.clip(min_y - 3, 0, h_img - 1)
        max_y = np.clip(max_y + 3, 0, h_img - 1)
        min_x = np.clip(min_x - max_w, 0, w_img - 1)
        max_x = np.clip(max_x + max_w * 2, 0, w_img - 1)

        # Extraer la región de la placa
        placa = intersection[min_y:max_y, min_x:max_x]

        # Encontrar componentes conectados
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(placa)
        output_placa = img_auto_rgb

        # Dibujar rectángulo alrededor de cada letra
        letras_encontradas =0
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            ratio = h / w
            area = w * h
            if  30 < area < 250 and 7 <= h <= 20 and letras_encontradas < 6:
              letras_encontradas += 1
              cv2.rectangle(output_placa, (min_x + x-1, min_y + y-2), (min_x + x + w+1, min_y + y + h+2), (0, 255,0), 1) 
        # Si no encontramos todas las letras de la patente probamos con otro tipo de treshhold
        if letras_encontradas < 6:
          placa = img_auto_gray[min_y:max_y, min_x:max_x]
          _,placa_tresh = cv2.threshold(placa, 140, 255, cv2.THRESH_TOZERO)
          n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(placa_tresh)
          output_placa = img_auto_rgb

          for i in range(1, n_labels):
              x, y, w, h, area = stats[i]
              ratio = h / w
              area = w * h
              if  30 < area < 250 and 7 <= h <= 20 :
                cv2.rectangle(output_placa, (min_x + x-1, min_y + y-2), (min_x + x + w+1, min_y + y + h+2), (0, 255,0), 1) 
                
    #Recuadramos la patente
    cv2.rectangle(output_placa, (min_x, min_y), (max_x, max_y), (255,0,0), 1)  
    imshow(output_placa,colorbar=False, color_img=True, blocking=True, title='Detección de patentes')