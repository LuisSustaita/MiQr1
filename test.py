def cambia_tamano_de_esta_imagen(nombre_imagen):
    from PIL import Image

    Width = 200
    Height = 200

    img = Image.open(nombre_imagen)

    imgresized = img.resize((Width, Height), Image.NEAREST)

    imgresized.save(nombre_imagen)


# Importamos OpenCV para realizar operaciones a las imegenes
import cv2
# Importamos qrcode para generar Codigos QR propios
import qrcode

Qr = qrcode.make("Que tranza carranza")
imagen = open("qrprueba.png", "wb")
Qr.save(imagen)
imagen.close()

cambia_tamano_de_esta_imagen('qrprueba.png')
cambia_tamano_de_esta_imagen('imagenprueba.jpg')

imagen1 = cv2.imread('qrprueba.png')
imagen2 = cv2.imread('imagenprueba.jpg')

imagen3 = imagen1[1:200, 1:200, :]

# fusion de imagenes
# suma de imagenes, pero con diferentes pesos lo cual da una
# sensacion de mezcla o transparencia
# toma el 0.7 de la primer imagen y el 0.3 de la segunda imagen
imagen_resultado = cv2.addWeighted(imagen3, 0.7, imagen2, 0.3, 0)
# gamma es tomado como cero.
pu

cv2.imshow('imagen_resultado', imagen_resultado)
cv2.imwrite('imagen_resultado.png', imagen_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
