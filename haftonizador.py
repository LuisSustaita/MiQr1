import os
import numpy as np
from PIL import Image
import cv2
from matricesGeneradas import matriz

def halftoning(imagen, matrizThresholds,nombrenuevo):

    img = cv2.imread(imagen)

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray = np.zeros((img.shape[0],img.shape[1]))

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img_gray[x][y] = int((img[x, y, 0] * 0.2126) + (img[x, y, 1] * 0.7152) + (img[x, y, 2] * 0.0722))

    for i in range(0,len(img_gray),len(matrizThresholds)):
        for j in range(0,len(img_gray[0]),len(matrizThresholds[0])):
            for x in range(i,i+len(matrizThresholds)):
                for y in range(j,j+len(matrizThresholds[0])):
                    try:
                        gris = img_gray[x][y]
                        threshold = matrizThresholds[x-i][y-j]
                        if gris > threshold:
                            img_gray[x][y]=255
                        else:
                            img_gray[x][y]=0
                    except:
                        pass

    pedacitos = imagen.split("/")[len(imagen.split("/")) - 1].split(".")
    cv2.imwrite("C:/Users/luisd/Desktop/tesis/imagenes/halftoneadas2/"+nombrenuevo+pedacitos[0]+'.jpg', np.array(img_gray))

    return img_gray

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

#-----
matrizThresholdG=matriz.matrizThresholdG9x9
matrizThresholdI=matriz.matrizThresholdI16x16
#-----

rutaNormal="C:/Users/luisd/Desktop/tesis/imagenes/normales/"


listing = os.listdir(rutaNormal)

for file in listing:
    cambia_tamano_de_esta_imagen(rutaNormal + file, 500)
    halftoned_image = halftoning(rutaNormal + file, matrizThresholdI, "nuevaI")

