# Importamos OpenCV para realizar operaciones a las imegenes
import cv2
# Importamos qrcode para generar Codigos QR propios
import qrcode
import numpy as np
from PIL import Image


def find_new_pixel(pixel):
    return 0 if pixel < 128 else 255


# Floyd and Steinberg https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf pag 22
def FloydDithering(imagen):
    img = cv2.imread(imagen)

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = []

    for x in range(img.shape[0]):
        img_gray.append([])
        for y in range(img.shape[1]):
            img_gray[x].append([])
            img_gray[x][y] = (img[x, y, 0] * 0.2126) + (img[x, y, 1] * 0.7152) + (img[x, y, 2] * 0.0722)

    matrizimagen = np.zeros((200, 200))

    for x in range(len(img_gray)-1):
        for y in range(len(img_gray[x])-1):
            oldpixel = img_gray[x][ y]
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
    return matrizimagen


def redimensionarQR(nombre_imagen,escala):
    """
    :param nombre_imagen: string con el nombre de la imagen a redimensionar
    :param escala: entero de cuantos pixeles se va a extender (ejemplo  de 1 pixel a 3 ... 1 -> 5)
    :return: objeto Image con la imagen redimensionada
    """

    img = Image.open(nombre_imagen)

    width, height = img.size

    Width = width*escala
    Height = height*escala

    imgresized = img.resize((Width, Height), Image.ANTIALIAS)

    imgresized.save("new "+nombre_imagen)

    return imgresized


def cambia_tamano_de_esta_imagen(nombre_imagen):
    """
    :param nombre_imagen:  string con el nombre de la imagen a redimensionar
    :return: objeto Image con la imagen redimensionada a 200*200 pixeles
    """
    img = Image.open(nombre_imagen)

    Width = 200
    Height = 200

    imgresized = img.resize((Width, Height), Image.ANTIALIAS)

    imgresized.save(nombre_imagen)

    return imgresized


def halftoneQR(QR, imagen):
    newQR= np.asarray(QR)
    for x in range(QR.shape[0]):
        for y in range(QR.shape[1]):
            if np.random.uniform(0,1) < 0.3:
                newQR[x, y] = QR[x, y]
            else:
                newQR[x, y] = imagen[x, y]

    return newQR


def binarizacionSimple(imagen, threshold):
    """
    :param imagen: string con el nombre de la imagen a binarizar
    :param threshold: entero entre 0 y 512 que representa el limite
    :return: imagen Binarizada
    """
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
    return img_gray


qr = qrcode.QRCode(
    version=10,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=1,
)

# 0 -17 caracteres -> 23*23 QR
# 18-32 caracteres -> 27*27 QR
# 33-53 caracteres -> 31*31 QR
# 54-78 caracteres -> 35*35 QR -> 0.4
# 79->¿ caracteres -> 39*39 QR
#                  -> 43*43
#                  -> 47*47
#qr.add_data('hey')
qr.add_data('BEGIN:VCARD\nVERSION:3.0\n'
            'N:Sustaita;Luis\nTEL:4491046191\n'
            'EMAIL:ld.delacruzsustaita@ugto.mx\n'
            '\nEND:VCARD')
qr.make(fit=True)

img = qr.make_image()

NOMBRE_DE_QR = "qrsito.png"

img.save(NOMBRE_DE_QR)

arrayQR = np.asarray(img, dtype=np.int)
for i in range(len(arrayQR)):
    for j in range(len(arrayQR[i])):
        if arrayQR[i][j] == 1:
            arrayQR[i][j] = 255
        else:
            arrayQR[i][j] = 0

imagen = 'pat2.jpeg'

QR_Redimensionado = redimensionarQR(NOMBRE_DE_QR, 3)

cambia_tamano_de_esta_imagen("new "+NOMBRE_DE_QR)
cambia_tamano_de_esta_imagen(imagen)

# binarizacion(imagen, 127)
# FloydDithering(imagen)
halftoned_image = JarvisDithering(imagen)

newQR = halftoneQR(arrayQR, halftoned_image)
#cv2.imwrite('NEW QR.jpg', newQR)

imagen1 = cv2.imread('qrsito.png')
imagen2 = cv2.imread(imagen)

# fusion de imagenes
# suma de imagenes, pero con diferentes pesos lo cual da una
# sensacion de mezcla o transparencia
# toma el 0.7 de la primer imagen y el 0.3 de la segunda imagen
imagen_resultado = cv2.addWeighted(imagen1, 0.3, imagen2, 0.7, 0)
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
# Deteccion de codigos QR
# https://robologs.net/2017/07/17/deteccion-de-codigos-qr-en-python-con-opencv-y-zbar/

# Grayscale Digital Halftoning using Optimization Techniques
# http://ethesis.nitrkl.ac.in/7814/1/2015_Grayscale_Lalitha.pdf
# Threshold matrix generation for digital halftoning by genetic algorithm optimizatio
# https://pdfs.semanticscholar.org/82e4/78f83f42c0a84528722a27e8d0d91f3fa3b8.pdf
# Halftone Image Generation with Improved Multiobjective Genetic Algorithm
# https://www.researchgate.net/publication/2366306_Halftone_Image_Generation_with_Improved_Multiobjective_Genetic_Algorithm
# Aesthetic QR code generation with background contrast enhancement and user interaction
# file:///C:/Users/luisd/Downloads/108280G.pdf
# Stylize Aesthetic QR Code
# https://arxiv.org/pdf/1803.01146.pdf
# Efficient QR Code Beautification With High Quality Visual Content
# http://graphics.csie.ncku.edu.tw/QR_code/QR_code_TMM.pdf
# ART-UP: A Novel Method for Generating Scanning-robust Aesthetic QR codes
# https://arxiv.org/pdf/1803.02280.pdf
# HALFTONED QR
# https://jsfiddle.net/lachlan/r8qWV/
#
# ALGORITMO GENETICO CON PENALIZACION DE MUTACIONES
# https://dam-prod.media.mit.edu/x/files/wp-content/uploads/sites/10/2013/07/spie97newbern.pdf
#
# DEFINIR UN OBJETO PIXEL CON LOS ATRIBUTOS DE COORDENADA(X,Y) Y NIVEL DE ILUMINACION
# DEFINIR UNA FUNCION QUE "ENCUENTRE" LOS MODULOS (CUADRO BLANCO O NEGRO) DEL QR Y SU TAMAÑO PARA DEFINIR LOS PIXELES CENTRALES
# DEFINIR UNA FUNCION QUE REGRESE LA MASCARA DE ACUERDO AL MODULO, (1 -> PIXEL(ES) CENTRAL(ES) DEL MODULO  0 -> NO ES PIXEL CENTRAL)

#