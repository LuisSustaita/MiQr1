from fileinput import filename

import qrcode
import numpy as np
from PIL import Image
import cv2
from matricesGeneradas import matriz


def find_new_pixel(pixel):
    return 0 if pixel < 128 else 255

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
    cv2.imwrite("C:/Users/luisd/Desktop/tesis/imagenes/halftoneadas/"+nombrenuevo+pedacitos[0]+'.jpg', np.array(img_gray))

    return img_gray

def JarvisDithering(imagen, tamano):
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

    pedacitos = imagen.split("/")[len(imagen.split("/")) - 1].split(".")
    cv2.imwrite("C:/Users/luisd/Desktop/tesis/imagenes/halftoneadas/"+'JarvisDit'+pedacitos[0]+'.jpg', matrizimagen)
    return matrizimagen

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


def UnosA255(arrayQR):
    for i in range(len(arrayQR)):
        for j in range(len(arrayQR[i])):
            if arrayQR[i][j] == 1:
                arrayQR[i][j] = 255
            else:
                arrayQR[i][j] = 0
    return arrayQR


def redimensionarQR(nombre_imagen, escala):
    """
    :param nombre_imagen: string con el nombre de la imagen a redimensionar
    :param escala: entero de cuantos pixeles se va a extender (ejemplo  de 1 pixel a 3 ... 1 -> 5)
    :return: objeto Image con la imagen redimensionada
    """

    img = Image.open(nombre_imagen)

    width, height = img.size

    Width = width * escala
    Height = height * escala

    imgresized = img.resize((Width, Height), Image.ANTIALIAS)

    imgresized.save(nombre_imagen.split(".")[0]+"new.png")

    return imgresized


def FindModuleSize(QR_Image):
    QR_array = cv2.imread(QR_Image)
    aux = QR_array[0, 0, 0]
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
                if cont1 < 2:
                    contador += 1
                else:
                    break
        if contador != 0:
            break
    return contador / 7


def FindPatterns(QR_Image):
    QR_array = cv2.imread(QR_Image)
    aux = QR_array[0, 0, 0]
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
                if cont1 < 2:
                    contador += 1
                else:
                    break
        if contador != 0:
            break

    for i in range(int(FindModuleSize(QR_Image)), contador + int(FindModuleSize(QR_Image))):
        for j in range(int(FindModuleSize(QR_Image)), contador + int(FindModuleSize(QR_Image))):
            QR_array[i][j] = [11, 2, 123]

    for i in range(len(QR_array) - contador - int(FindModuleSize(QR_Image)),
                   len(QR_array) - int(FindModuleSize(QR_Image))):
        for j in range(int(FindModuleSize(QR_Image)),
                       contador + int(FindModuleSize(QR_Image))):
            QR_array[i][j] = [11, 2, 123]

    for i in range(int(FindModuleSize(QR_Image)),
                   contador + int(FindModuleSize(QR_Image))):
        for j in range(len(QR_array) - contador - int(FindModuleSize(QR_Image)),
                       len(QR_array) - int(FindModuleSize(QR_Image))):
            QR_array[i][j] = [11, 2, 123]

    return QR_array, [[int(FindModuleSize(QR_Image)), contador + int(FindModuleSize(QR_Image))],
                      [len(QR_array) - contador - int(FindModuleSize(QR_Image)),
                       len(QR_array) - int(FindModuleSize(QR_Image))]]


def CentralPixelMask(QR_image):
    # 0 -> no es pixel central
    # 1 -> es pixel central
    Tamano_del_modulo = int(FindModuleSize(QR_image))
    QR_array = cv2.imread(QR_image)

    maskaredQR_array = np.zeros((len(QR_array), len(QR_array[0])))

    BorderSize = Tamano_del_modulo // 3

    reemplazo = np.ones((Tamano_del_modulo - (BorderSize * 2), Tamano_del_modulo - (BorderSize * 2)))

    reemplazo = np.pad(reemplazo, BorderSize, mode='constant', constant_values=0)

    for i in range(0, len(QR_array), Tamano_del_modulo):
        for j in range(0, len(QR_array[i]), Tamano_del_modulo):
            # if QR_array[i][j][0] !=0:
            aux1 = 0
            for k in range(i, i + Tamano_del_modulo):
                aux2 = 0
                for m in range(j, j + Tamano_del_modulo):
                    maskaredQR_array[k][m] = reemplazo[aux1][aux2]
                    aux2 += 1
                aux1 += 1

    return maskaredQR_array


def QR_IN_IMAGE(qr_imagen, imagen, centralpixels_array, findpaterns):
    qr_array = cv2.imread(qr_imagen)
    imagen_array = cv2.imread(imagen)

    for i in range(findpaterns[0][0], findpaterns[0][1]):
        for j in range(findpaterns[0][0]+1, findpaterns[0][1]):
            imagen_array[i][j] = qr_array[i][j]

    for i in range(findpaterns[0][0], findpaterns[0][1]):
        for j in range(findpaterns[1][0], findpaterns[1][1]):
            imagen_array[i][j] = qr_array[i][j]

    for i in range(findpaterns[1][0], findpaterns[1][1]):
        for j in range(findpaterns[0][0], findpaterns[0][1]):
            imagen_array[i][j] = qr_array[i][j]

    for i in range(len(qr_array)):
        for j in range(len(qr_array[i])):
            if centralpixels_array[i][j] == 255:
                imagen_array[i][j] = qr_array[i][j]

    cv2.imwrite(""+imagen.split("/")[len(imagen.split("/")) - 1].split(".")[0]+'Antes.jpg', imagen_array)
    cv2.imshow("image", imagen_array)

    #Image.fromarray(imagen_array, 'RGB').show()


def QR_IN_IMAGE2(qr_imagen, imagen, centralpixels_array, findpaterns, concentracion_de_puntos, tamano_de_modulo, filtro):

    qr_array = cv2.imread(qr_imagen)
    imagen_array = cv2.imread(imagen)

    i_modulo = 0
    imagen_array2 = imagen_array
    for i in range(len(qr_array)):
        j_modulo = 0
        for j in range(len(qr_array[i])):

            if centralpixels_array[i][j] == 255:
                imagen_array[i][j] = [qr_array[i][j][0]*0.55 + 0.45*imagen_array2[i][j][0],
                                      qr_array[i][j][1]*0.55 + 0.45*imagen_array2[i][j][1],
                                      qr_array[i][j][2]*0.55 + 0.45*imagen_array2[i][j][2]]
            else:

                maximo_puntaje= tamano_de_modulo*tamano_de_modulo
                porcentaje = concentracion_de_puntos[i_modulo][j_modulo] * .2 / maximo_puntaje

                resto = 1-porcentaje

                imagen_array[i][j] = [(imagen_array2[i][j][0] * resto) + (porcentaje * qr_array[i][j][0]),
                                      (imagen_array2[i][j][1] * resto) + (porcentaje * qr_array[i][j][1]),
                                      (imagen_array2[i][j][2] * resto) + (porcentaje * qr_array[i][j][2])]

            if (j+1) % tamano_de_modulo == 0:
                j_modulo += 1
        if (i+1) % tamano_de_modulo == 0:
            i_modulo +=1

    for i in range(findpaterns[0][0], findpaterns[0][1]):
        for j in range(findpaterns[0][0]+1, findpaterns[0][1]):
            imagen_array[i][j] = qr_array[i][j]

    for i in range(findpaterns[0][0], findpaterns[0][1]):
        for j in range(findpaterns[1][0], findpaterns[1][1]):
            imagen_array[i][j] = qr_array[i][j]

    for i in range(findpaterns[1][0], findpaterns[1][1]):
        for j in range(findpaterns[0][0], findpaterns[0][1]):
            imagen_array[i][j] = qr_array[i][j]

    cv2.imwrite(""+imagen.split("/")[len(imagen.split("/")) - 1].split(".")[0]+filtro+'Despues.jpg', imagen_array)
    cv2.imshow("image2", imagen_array)

    #Image.fromarray(imagen_array, 'RGB').show()


def ColorHalftone(halftonemask_array,imagen):
    imagen_array = cv2.imread(imagen)

    for i in range(len(halftonemask_array)):
        for j in range(len(halftonemask_array[i])):
            if halftonemask_array[i][j] == 255:
                imagen_array[i][j] = 255

    cv2.imwrite('colorhalftone.jpg', imagen_array)
    cv2.imshow("imagech", imagen_array)

    #Image.fromarray(imagen_array, 'RGB').show()


def Puntos_por_modulo(halftoned_image,tamano_de_modulo):
    """
    :param halftoned_image: matriz de numpy
    :param module_mask:  matriz de numpy
    :return:
    """
    matriz_de_concentracion=[]
    for i in range(0,len(halftoned_image),tamano_de_modulo):
        for j in range(0,len(halftoned_image[i]),tamano_de_modulo):
            contador_de_puntos=0
            for k in range(i,i+tamano_de_modulo):
                for m in range(j,j+tamano_de_modulo):
                    if halftoned_image[k][m] == 0:
                        contador_de_puntos+=1
            matriz_de_concentracion.append(contador_de_puntos)

    return matriz_de_concentracion

# ***** Crear QR
borde = 1
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=1,
    border=borde,
)
#qr.add_data('BEGIN:VCARD\nVERSION:3.0\n'+
#            'N:Sustaita;Luis\nTEL:4491046191\n'+
#            '\nEND:VCARD')

qr.add_data('http://bit.ly/chuchitafan')
imagen = '22b.jpg'
# Genetico
filtro="Gen"
# Iterated Local Search
#filtro = "ILS"


qr.make(fit=True)
img = qr.make_image()
rutaQRs=""
NOMBRE_DE_QR = "TESTqrsito.png"
img.save(rutaQRs+NOMBRE_DE_QR)
# **********

#-----
matrizThresholdG=matriz.matrizThresholdG9x9
matrizThresholdI=matriz.matrizThresholdI16x16
#-----

# ----
arrayQR = np.asarray(img, dtype=np.int)

arrayQR = UnosA255(arrayQR)

QR_Redimensionado = redimensionarQR(rutaQRs+NOMBRE_DE_QR, 9)

# -------- Crea mascara de pixeles centrales de QR
maskaredModule = CentralPixelMask(rutaQRs+ NOMBRE_DE_QR.split(".")[0]+"new.png")  # Mascara de pixeles centrales matriz numpy (0 - 1)
maskaredModule255 = UnosA255(maskaredModule)  # Mascara de pixeles centrales matriz numpy (0 - 255)              <-----
# --------------------------
show2 = Image.fromarray(maskaredModule255)
show2.show()

#Qr_array = FindPatterns("new " + NOMBRE_DE_QR)[0]
#print(FindPatterns("new "+NOMBRE_DE_QR)[1])
#Image.fromarray(Qr_array).show()

tamano = maskaredModule.shape[0]
rutaNormal=""
cambia_tamano_de_esta_imagen(rutaNormal+imagen, tamano)

rutaHalf = ""
# JarvisDithering
#filtro="Jarvis"
#halftoned_image = JarvisDithering(rutaNormal+imagen, tamano)

if filtro=="ILS":
    halftoned_image = halftoning(rutaNormal+imagen,matrizThresholdI,"nuevaI")
elif filtro == "Gen":
    halftoned_image = halftoning(rutaNormal + imagen, matrizThresholdG, "nuevaG")


#ColorHalftone(halftoned_image,imagen)
Image.fromarray(halftoned_image).show()
QR_IN_IMAGE(rutaQRs+NOMBRE_DE_QR.split(".")[0]+"new.png", rutaNormal+imagen, maskaredModule255, FindPatterns(rutaQRs+NOMBRE_DE_QR.split(".")[0]+"new.png")[1])

tamano_de_modulo = int(FindModuleSize(rutaQRs+NOMBRE_DE_QR.split(".")[0]+"new.png"))
concentracion_de_puntos = np.array(Puntos_por_modulo(halftoned_image,tamano_de_modulo))
concentracion_de_puntos = concentracion_de_puntos.reshape(int(np.sqrt(len(concentracion_de_puntos))),int(np.sqrt(len(concentracion_de_puntos))))
j=Image.fromarray(concentracion_de_puntos)
j.show()
j.save("concentracion.png")

QR_IN_IMAGE2(rutaQRs+NOMBRE_DE_QR.split(".")[0]+"new.png", rutaNormal+imagen, maskaredModule255, FindPatterns(rutaQRs+NOMBRE_DE_QR.split(".")[0]+"new.png")[1],concentracion_de_puntos,tamano_de_modulo,filtro)
cv2.waitKey()
