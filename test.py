def cambia_tamano_de_esta_imagen(nombre_imagen):
    from PIL import Image

    Width = 200
    Height = 200

    img = Image.open(nombre_imagen)

    imgresized = img.resize((Width, Height), Image.NEAREST)

    imgresized.save(nombre_imagen)


def binarizacion(imagen, threshold):
    import cv2
    img = cv2.imread(imagen)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('gray.jpg', img_gray)

    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if img_gray[i, j] > threshold:
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 255

        cv2.imwrite('bin.jpg', img_gray)


# Importamos OpenCV para realizar operaciones a las imegenes
import cv2
# Importamos qrcode para generar Codigos QR propios
import qrcode

Qr = qrcode.make("Que tranza carranza")
imagen = open("qrprueba.png", "wb")
Qr.save(imagen)
imagen.close()

cambia_tamano_de_esta_imagen('qrprueba.png')
cambia_tamano_de_esta_imagen('carrito.png')

binarizacion('carrito.png',155)

imagen1 = cv2.imread('qrprueba.png')
imagen2 = cv2.imread('carrito.png')

# imagen3 = imagen1[1:200, 1:200, :]

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
