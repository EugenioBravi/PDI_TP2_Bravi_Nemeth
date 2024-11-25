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

def clasificar_monedas_y_dados(imagen):
    
    # Convertimos la imagen a RGB para mostrar los resultados
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    # Convertimos la imagen a escala de grises para la detección de círculos
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Clasificación de monedas
    contador_monedas = {'monedas_10_centavos': 0, 'monedas_50_centavos': 0, 'monedas_1_peso': 0}
    
    # Detectamos círculos (monedas) usando la Transformada de Hough
    monedas_detectadas = cv2.HoughCircles(imagen_gris, cv2.HOUGH_GRADIENT, 1.4, minDist=100, param1=90, param2=190, minRadius=100, maxRadius=200)
    monedas_detectadas = np.uint16(np.around(monedas_detectadas))

    # Procesamos cada moneda detectada
    for moneda in monedas_detectadas[0]:
        centro_x = moneda[0]
        centro_y = moneda[1]
        radio = moneda[2]

        # Clasificamos el tipo de moneda según el radio detectado
        if radio >= 167:
            contador_monedas['monedas_50_centavos'] += 1
            color = (0, 0, 255)  # Rojo para moneda de 50 centavos
        elif radio >= 159:
            contador_monedas['monedas_1_peso'] += 1
            color = (255, 0, 0)  # Azul para moneda de 1 peso
        else:
            contador_monedas['monedas_10_centavos'] += 1
            color = (0, 255, 0)  # Verde para moneda de 10 centavos

        # Dibujamos un círculo alrededor de la moneda detectada
        cv2.circle(imagen_rgb, (centro_x, centro_y), radio, color, 4)

    # Clasificación de dados
    # Detectamos círculos más pequeños (dados) usando la Transformada de Hough
    dados_detectados = cv2.HoughCircles(imagen_gris, cv2.HOUGH_GRADIENT, 1, minDist=10, param1=50, param2=50, minRadius=5, maxRadius=30)
    dados_detectados = np.uint16(np.around(dados_detectados))

    # Dividimos la imagen en dos mitades para contar los dados en cada lado
    mitad_ancho_imagen = imagen.shape[1] // 2
    contador_dados = [0, 0] 

    # Procesamos cada dado detectado
    for dado in dados_detectados[0]:
        centro_x = dado[0]
        centro_y = dado[1]
        radio = dado[2]

        # Clasificamos el dado según su posición en la imagen (izquierda o derecha)
        if centro_x < mitad_ancho_imagen:
            color = (0, 255, 255)  # Turquesa para el dado izquierdo
            contador_dados[0] += 1
        else:
            color = (255, 0, 255)  # Rosa para el dado derecho
            contador_dados[1] += 1

        # Dibujamos un círculo alrededor del dado detectado
        cv2.circle(imagen_rgb, (centro_x, centro_y), radio, color, 6)

    return contador_monedas, contador_dados, imagen_rgb


# Leemos la imagen desde el archivo
imagen_entrada = cv2.imread('./img/monedas.jpg')

# Clasificamos las monedas y los dados
monedas, dados, imagen_clasificada = clasificar_monedas_y_dados(imagen_entrada)

monedas_10c = monedas['monedas_10_centavos']
monedas_50c = monedas['monedas_50_centavos']
monedas_1p = monedas['monedas_1_peso']

dado_izquierdo = dados[0]
dado_derecho = dados[1]

print(f"Hay {monedas_10c} monedas de 10 centavos, {monedas_50c} monedas de 50 centavos y {monedas_1p} monedas de 1 peso.")
print(f"El dado izquierdo muestra {dado_izquierdo} puntos y el dado derecho muestra {dado_derecho} puntos.")
imshow(imagen_clasificada,colorbar=False, color_img=True, blocking=True, title='Clasificación de monedas y dados')

