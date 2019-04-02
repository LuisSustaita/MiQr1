# Importamos OpenCV para realizar operaciones a las imegenes
import cv2
# Importamos qrcode para generar Codigos QR propios
import qrcode
import numpy as np


def find_new_pixel(pixel):
    return 0 if pixel < 128 else 255


# Floyd and Steinberg https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf pag 22
def FloydDithering(imagen):
    img = cv2.imread(imagen)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matrizimagen = np.zeros((200, 200))

    for x in range(img_gray.shape[0]-1):
        for y in range(img_gray.shape[1]-1):
            oldpixel = img_gray[x, y]
            newpixel = find_new_pixel(oldpixel)
            quant_error = oldpixel-newpixel

            part = quant_error / 16

            img_gray[x+1]  [y] = img_gray[x+1][y]   + (part * 7)
            img_gray[x-1][y+1] = img_gray[x-1][y+1] + (part * 3)
            img_gray[x]  [y+1] = img_gray[x]  [y+1] + (part * 5)
            img_gray[x+1][y+1] = img_gray[x+1][y+1] + (part * 1)

            matrizimagen[x][y] = newpixel

    cv2.imwrite('floydDit.jpg', matrizimagen)


# Jarvis, Judice, and Ninke https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf pag 22
def JarvisDithering(imagen):
    img = cv2.imread(imagen)

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = []

    for x in range(img.shape[0]):
        img_gray.append([])
        for y in range(img.shape[1]):
            img_gray[x].append([])
            img_gray[x][y] = (img[x, y, 0] * 0.2126) + (img[x, y, 1] * 0.7152) + (img[x, y, 2] * 0.0722)

    matrizimagen = np.zeros((200, 200))

    for x in range(len(img_gray) - 2):
        for y in range(len(img_gray[x]) - 2):
            oldpixel = img_gray[x][y]
            newpixel = find_new_pixel(oldpixel)
            quant_error = oldpixel - newpixel

            part = quant_error / 48

            img_gray[x + 1][ y] = round(img_gray[x + 1][ y] + part * 7)
            img_gray[x + 2][ y] = round(img_gray[x + 2][ y] + part * 5)

            img_gray[x - 2][ y + 1] = round(img_gray[x - 2][ y + 1] + part * 3)
            img_gray[x - 1][ y + 1] = round(img_gray[x - 1][ y + 1] + part * 5)
            img_gray[x][ y + 1] = round(img_gray[x][ y + 1] + part * 7)
            img_gray[x + 1][ y + 1] = round(img_gray[x + 1][ y + 1] + part * 5)
            img_gray[x + 2][ y + 1] = round(img_gray[x + 2][ y + 1] + part * 3)

            img_gray[x - 2][ y + 2] = round(img_gray[x - 2][ y + 2] + part * 1)
            img_gray[x - 1][ y + 2] = round(img_gray[x - 1][ y + 2] + part * 3)
            img_gray[x][ y + 2] = round(img_gray[x][ y + 2] + part * 5)
            img_gray[x + 1][ y + 2] = round(img_gray[x + 1][ y + 2] + part * 3)
            img_gray[x + 2][ y + 2] = round(img_gray[x + 2][ y + 2] + part * 1)

            matrizimagen[x][y] = newpixel

    cv2.imwrite('JarvisDit.jpg', matrizimagen)

def cambia_tamano_de_esta_imagen(nombre_imagen):
    from PIL import Image

    Width = 200
    Height = 200

    img = Image.open(nombre_imagen)

    imgresized = img.resize((Width, Height), Image.NEAREST)

    imgresized.save(nombre_imagen)


def binarizacion(imagen, threshold):
    img = cv2.imread(imagen)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('graybin.jpg', img_gray)

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            print(" pixel {}".format(img_gray[i, j]))
            if img_gray[i, j] > threshold:
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 255

    cv2.imwrite('bin'+str(threshold)+'.jpg', img_gray)


Qr = qrcode.make("Que tranza carranza")
qrimagen = open("qrprueba.png", "wb")
Qr.save(qrimagen)
qrimagen.close()
imagen = 'carrito.png'

cambia_tamano_de_esta_imagen('qrprueba.png')
cambia_tamano_de_esta_imagen(imagen)

# binarizacion(imagen, 127)
# FloydDithering(imagen)
JarvisDithering(imagen)

imagen1 = cv2.imread('qrprueba.png')
imagen2 = cv2.imread(imagen)

# fusion de imagenes
# suma de imagenes, pero con diferentes pesos lo cual da una
# sensacion de mezcla o transparencia
# toma el 0.7 de la primer imagen y el 0.3 de la segunda imagen
imagen_resultado = cv2.addWeighted(imagen1, 0.7, imagen2, 0.3, 0)
# gamma es tomado como cero

cv2.imshow('imagen_resultado', imagen_resultado)
cv2.imwrite('imagen_resultado.png', imagen_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Links de posible ayuda
# ARTICULO BASE
# file:///C:/Users/luisd/Downloads/DoubleColumn.pdf
# 26 28 30
# Presentacion basica
# https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf
# Presentacion no tan basica
# http://webstaff.itn.liu.se/~sasgo26/Dig_Halftoning/Lectures/Lecture_2013_1.pdf
# Tipos de esparcimiento
# https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/
# Libro de Halftoning
# file:///C:/Users/luisd/Downloads/epdf.tips_modern-digital-halftoning-second-edition-signal-pr.pdf
# Articulo medio pedorro de halftoning
# https://pdfs.semanticscholar.org/7f8f/9fed14f0bd509af776a90cb220f9c26ccb81.pdf
# TESIS Advanced halftoning methods
# file:///C:/Users/luisd/Downloads/BPTX_2008_2_11320_0_233062_0_73868.pdf
# Articulo A Simple and Efficient Error-Diffusion Algorithm
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8947&rep=rep1&type=pdf
#
