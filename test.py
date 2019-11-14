# Importamos OpenCV para realizar operaciones a las imegenes
import cv2
# Importamos qrcode para generar Codigos QR propios
import qrcode
import numpy as np
from PIL import Image


def UnosA255(arrayQR):
    for i in range(len(arrayQR)):
        for j in range(len(arrayQR[i])):
            if arrayQR[i][j] == 1:
                arrayQR[i][j] = 255
            else:
                arrayQR[i][j] = 0
    return arrayQR


def find_new_pixel(pixel):
    return 0 if pixel < 128 else 255


# Floyd and Steinberg https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf pag 22
def FloydDithering(imagen,nombre, tamano):
    img = cv2.imread(imagen)

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = []

    for x in range(img.shape[0]):
        img_gray.append([])
        for y in range(img.shape[1]):
            img_gray[x].append([])
            img_gray[x][y] = (img[x, y, 0] * 0.2126) + (img[x, y, 1] * 0.7152) + (img[x, y, 2] * 0.0722)

    matrizimagen = np.zeros((tamano, tamano))

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

    cv2.imwrite('C:/Users/luisd/Dropbox/tesis/codigos/IntentoQr1/Fin'+nombre, matrizimagen)
    return matrizimagen


# Jarvis, Judice, and Ninke https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf pag 22
def JarvisDithering(imagen, tamano, nombre):
    img = cv2.imread(imagen)

    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = []

    for x in range(img.shape[0]):
        img_gray.append([])
        for y in range(img.shape[1]):
            img_gray[x].append([])
            img_gray[x][y] = (img[x, y, 0] * 0.2126) + (img[x, y, 1] * 0.7152) + (img[x, y, 2] * 0.0722)

    matrizimagen = np.zeros((tamano, tamano))

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

    cv2.imwrite('C:/Users/luisd/Dropbox/tesis/codigos/IntentoQr1/Fin'+nombre, matrizimagen)
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


def cambia_tamano_de_esta_imagen(nombre_imagen, tamano):
    """
    :param nombre_imagen:  string con el nombre de la imagen a redimensionar
    :param tamano: integer cantidad para redimensionar
    :return: objeto Image con la imagen redimensionada al tamaño indicado
    """
    img = Image.open(nombre_imagen)

    Width = tamano
    Height = tamano

    imgresized = img.resize((Width, Height), Image.ANTIALIAS)

    imgresized.save(nombre_imagen)

    return imgresized


def FindModuleSize(QR_Image):
    QR_array = cv2.imread(QR_Image)
    aux = QR_array[0,0,0]
    flag1 = False
    cont1 = 0
    contador = 0
    for i in range(len(QR_array)):
        for pixel1 in QR_array[i]:
            if pixel1[0] != aux:
                aux = pixel1[0]
                cont1 += 1
                flag1 = True
            if flag1:
                if cont1<2:
                    contador += 1
                else:
                    break
        if contador != 0:
            break
    return contador/7


def CentralPixelMask(QR_image):
    # 0 -> no es pixel central
    # 1 -> es pixel central
    Tamano_del_modulo = int(FindModuleSize(QR_image))
    QR_array = cv2.imread(QR_image)

    maskaredQR_array = np.zeros((len(QR_array),len(QR_array[0])))

    BorderSize = Tamano_del_modulo//3

    reemplazo = np.ones((Tamano_del_modulo - (BorderSize * 2), Tamano_del_modulo - (BorderSize * 2)))

    reemplazo = np.pad(reemplazo, BorderSize, mode='constant', constant_values=0)

    for i in range(0, len(QR_array), Tamano_del_modulo):
        for j in range(0, len(QR_array[i]), Tamano_del_modulo):
            if QR_array[i][j][0] !=0:
                aux1=0
                for k in range(i,i+Tamano_del_modulo):
                    aux2=0
                    for m in range(j,j+Tamano_del_modulo):
                        maskaredQR_array[k][m]=reemplazo[aux1][aux2]
                        aux2+=1
                    aux1+=1

    return maskaredQR_array


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


def IluminationLevel(HalftoningMask, CentralPixelMask, QR_Image):
    """
    :return: matriz con el nivel de iluminacion para cada pixel
    Beta  0 -> if centralPixel = 0 , QR = 1, HalftoneMask = 1
    Alfa  1 -> if centralPiel = 0 , QR = 0, HalftoneMask = 1
    Betac 2 -> if centralPixel = 1 , QR = 1, HalftoneMask = 0
    Alfac 3 -> if centralPixel = 1 , QR = 0, HalftoneMask = 0
    Nada  4 -> otra cosa
    """

    Iluminati = np.zeros((len(QR_Image),len(QR_Image)))

    for i in range(len(Iluminati)):
        for j in range(len(Iluminati[i])):
            if CentralPixelMask[i][j] == 0 and (QR_Image[i][j].all() == 1 or QR_Image[i][j].all() == 255) and (HalftoningMask[i][j] == 1 or HalftoningMask[i][j] == 255):
                Iluminati[i][j] = 0
            elif CentralPixelMask[i][j] == 0 and QR_Image[i][j].all() == 0 and (HalftoningMask[i][j] == 1 or HalftoningMask[i][j] == 255):
                Iluminati[i][j] = 1
            elif (CentralPixelMask[i][j] == 1 or CentralPixelMask[i][j] == 255) and (QR_Image[i][j].all() == 1 or QR_Image[i][j].all() == 255) and HalftoningMask[i][j] == 0:
                Iluminati[i][j] = 2
            elif (CentralPixelMask[i][j] == 1 or CentralPixelMask[i][j] == 255) and QR_Image[i][j].all() == 0 and HalftoningMask[i][j] == 0:
                Iluminati[i][j] = 3
            else:
                Iluminati[i][j] = 4

    return Iluminati


# ********** Crear QR y obtener datos de el *********
qr = qrcode.QRCode(
    version=3,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1,
    border=1,
)

qr.add_data('BEGIN:VCARD\nVERSION:3.0\n'
            'N:Sustaita;Luis\nTEL:4491046191\n'
            'EMAIL:ld.delacruzsustaita@ugto.mx\n'
            '\nEND:VCARD')
qr.make(fit=True)
img = qr.make_image()
NOMBRE_DE_QR = "qrsito.png"
img.save(NOMBRE_DE_QR)
QR_Redimensionado = redimensionarQR(NOMBRE_DE_QR, 6) # QR objeto Image nuevo tamaño
arrayQR = cv2.imread("new "+NOMBRE_DE_QR) # QR matriz numpy (0 - 255)                                           <------
Tamano_del_modulo = FindModuleSize("new "+NOMBRE_DE_QR) # integer con el tamaño del modulo
# ********************
show1 = Image.fromarray(arrayQR)
#show1.show()

# -------- Crea mascara de pixeles centrales de QR
maskaredModule = CentralPixelMask("new "+NOMBRE_DE_QR) # Mascara de pixeles centrales matriz numpy (0 - 1)
maskaredModule255 = UnosA255(maskaredModule) # Mascara de pixeles centrales matriz numpy (0 - 255)              <-----
# --------------------------
show2 = Image.fromarray(maskaredModule255)
#show2.show()

# Obtener tamaño de QR para redimensionar la imagen a ese tamaño
tamano = maskaredModule.shape[0]
# --------------------------

# ---------- Crear imagen con Halftone
print("Creando Halftone...")
nombre = "ardilla2.jpeg"
imagen = 'C:/Users/luisd/Dropbox/tesis/imagenes/normales/'+nombre
cambia_tamano_de_esta_imagen(imagen, tamano)
halftoned_image = FloydDithering(imagen, nombre, tamano) # imagen Halftoneada ggg matriz de numpy (0 - 255)            <-------
# --------------------------
show3 = Image.fromarray(halftoned_image)
show3.show()

# ********* Crea matriz de iluminacion
#print("Creando matriz de iluminación...")
#LuminanceLevel = IluminationLevel(halftoned_image,maskaredModule255,arrayQR)


"""
#CODIGO ANTIGUO
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
"""

# Links de posible ayuda

#R. W. Floyd and L. Steinberg. An adaptive algorithm for spatial X
#gray-scale. Proceedings Society Information Display, 17(2):75–78,
#1976.

#J. F. Jarvis, C. N. Judice, and W. H. Ninke. A survey of techniques for the display X
#of continuous tone pictures on bilevel displays. Computer Graphics and Image
#Processing, 5:13–40, 1976.

# Halftoning review and analysis
# http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S1692-33242012000200014 X
# ARTICULO BASE
# file:///C:/Users/luisd/Downloads/DoubleColumn.pdf X
# 26 28 30
# Presentacion basica
# https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf X
# Presentacion no tan basica
# http://webstaff.itn.liu.se/~sasgo26/Dig_Halftoning/Lectures/Lecture_2013_1.pdf X
# Tipos de esparcimiento
# https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/ X
# Libro de Halftoning
# file:///C:/Users/luisd/Downloads/epdf.tips_modern-digital-halftoning-second-edition-signal-pr.pdf X
# Articulo medio pedorro de halftoning
# https://pdfs.semanticscholar.org/7f8f/9fed14f0bd509af776a90cb220f9c26ccb81.pdf X
# TESIS Advanced halftoning methods
# file:///C:/Users/luisd/Downloads/BPTX_2008_2_11320_0_233062_0_73868.pdf X
# Articulo A Simple and Efficient Error-Diffusion Algorithm
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.8947&rep=rep1&type=pdf X
# Deteccion de codigos QR
# https://robologs.net/2017/07/17/deteccion-de-codigos-qr-en-python-con-opencv-y-zbar/

# Grayscale Digital Halftoning using Optimization Techniques
# http://ethesis.nitrkl.ac.in/7814/1/2015_Grayscale_Lalitha.pdf X
# Threshold matrix generation for digital halftoning by genetic algorithm optimizatio
# https://pdfs.semanticscholar.org/82e4/78f83f42c0a84528722a27e8d0d91f3fa3b8.pdf X
# Halftone Image Generation with Improved Multiobjective Genetic Algorithm
# https://www.researchgate.net/publication/2366306_Halftone_Image_Generation_with_Improved_Multiobjective_Genetic_Algorithm C
# Aesthetic QR code generation with background contrast enhancement and user interaction
# file:///C:/Users/luisd/Downloads/108280G.pdf X
# Stylize Aesthetic QR Code
# https://arxiv.org/pdf/1803.01146.pdf X
# Efficient QR Code Beautification With High Quality Visual Content
# http://graphics.csie.ncku.edu.tw/QR_code/QR_code_TMM.pdf X
# ART-UP: A Novel Method for Generating Scanning-robust Aesthetic QR codes
# https://arxiv.org/pdf/1803.02280.pdf X
# HALFTONED QR
# https://jsfiddle.net/lachlan/r8qWV/
# DECODIFICACION DE QR
# https://www.fing.edu.uy/inco/proyectos/butia/mediawiki/images/b/ba/Analisis_decodificaci%C3%B3n_QR_Code.pdf
#
# ALGORITMO GENETICO CON PENALIZACION DE MUTACIONES
# https://dam-prod.media.mit.edu/x/files/wp-content/uploads/sites/10/2013/07/spie97newbern.pdf X
#
#
#PATENTE
#https://patentimages.storage.googleapis.com/45/23/85/c349724f4dcf9d/US5726435.pdf X
#ISO
#file:///C:/Users/luisd/Downloads/ISO%20IEC%2018004%202015%20Standard.pdf
#