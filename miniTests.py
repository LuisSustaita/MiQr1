import qrcode
import numpy as np
from PIL import Image
import cv2

def UnosA255(arrayQR):
    for i in range(len(arrayQR)):
        for j in range(len(arrayQR[i])):
            if arrayQR[i][j] == 1:
                arrayQR[i][j] = 255
            else:
                arrayQR[i][j] = 0
    return arrayQR

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

def IluminationLevel(HalftoningMask, CentralPixelMask, QR_Image):
    """
    :return: matriz con el nivel de iluminacion para cada pixel
    Beta  0 -> if centralPixel = 0 , QR = 1, HalftoneMask = 1
    Alfa  1 -> if centralPixel = 0 , QR = 0, HalftoneMask = 1
    Betac 2 -> if centralPixel = 1 , QR = 1, HalftoneMask = 0
    Alfac 3 -> if centralPixel = 1 , QR = 0, HalftoneMask = 0
    Nada  4 -> otra cosa
    """
# ***** Crear QR
qr = qrcode.QRCode(
    version=10,
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
NOMBRE_DE_QR = "TESTqrsito.png"
img.save(NOMBRE_DE_QR)
# **********

# ----
arrayQR = np.asarray(img, dtype=np.int)

arrayQR = UnosA255(arrayQR)

QR_Redimensionado = redimensionarQR(NOMBRE_DE_QR, 6)

Tamano_del_modulo = FindModuleSize("new "+NOMBRE_DE_QR)

print("Tamaño del modulo "+str(Tamano_del_modulo))

maskaredModule = CentralPixelMask("new "+NOMBRE_DE_QR)

maskaredModule = UnosA255(maskaredModule)

tamano = maskaredModule.shape[0]
print(tamano)

showmaskared = Image.fromarray(maskaredModule)

showmaskared.show()


