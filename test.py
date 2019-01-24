# Importamos OpenCV para realizar operaciones a las imegenes
import cv2
# Importamos qrcode para generar Codigos QR propios
import qrcode


def find_closest_palette_color(pixel):
    return round(pixel/255)


def FloydDithering(imagen):
    img = cv2.imread(imagen)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for x in range(img_gray.shape[0]-1):
        for y in range(img_gray.shape[1]-1):
            oldpixel = img_gray[x, y]
            newpixel = find_closest_palette_color(oldpixel)
            img_gray[x, y] = newpixel
            quant_error = oldpixel-newpixel

            img_gray[x+1, y  ] = round(img_gray[x+1, y  ]+quant_error * 7/16)
            img_gray[x-1, y+1] = round(img_gray[x-1, y+1]+quant_error * 3/16)
            img_gray[x  , y+1] = round(img_gray[x  , y+1]+quant_error * 5/16)
            img_gray[x+1, y+1] = round(img_gray[x+1, y+1]+quant_error * 1/16)
    cv2.imwrite('floydDit.jpg', img_gray)
    binarizacion('floydDit.jpg', 23)


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

    cv2.imwrite('gray.jpg', img_gray)

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if img_gray[i, j] > threshold:
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 255

    cv2.imwrite('bin'+str(threshold)+'.jpg', img_gray)

# def luminance_Beta(QR, imagen_original):
# def luminance_Alfa(QR, imagen_original):
# def luminance_BetaC(QR, imagen_original):
# def luminance_AlfaC(QR, imagen_original):


Qr = qrcode.make("Que tranza carranza")
imagen = open("qrprueba.png", "wb")
Qr.save(imagen)
imagen.close()

cambia_tamano_de_esta_imagen('qrprueba.png')
cambia_tamano_de_esta_imagen('carrito.png')

#binarizacion('carrito.png', 127)
FloydDithering('carrito.png')

imagen1 = cv2.imread('qrprueba.png')
imagen2 = cv2.imread('carrito.png')

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
# https://engineering.purdue.edu/~bouman/ece637/notes/pdf/Halftoning.pdf
# http://webstaff.itn.liu.se/~sasgo26/Dig_Halftoning/Lectures/Lecture_2013_1.pdf
# https://blog.demofox.org/2017/10/20/generating-blue-noise-sample-points-with-mitchells-best-candidate-algorithm/
# file:///C:/Users/luisd/Downloads/epdf.tips_modern-digital-halftoning-second-edition-signal-pr.pdf
# https://pdfs.semanticscholar.org/7f8f/9fed14f0bd509af776a90cb220f9c26ccb81.pdf
# file:///C:/Users/luisd/Downloads/BPTX_2008_2_11320_0_233062_0_73868.pdf
