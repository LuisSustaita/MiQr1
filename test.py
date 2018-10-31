# Importamos OpenCV para realizar operaciones a las imegenes
import cv2

imagen1 = cv2.imread('qrprueba.png')
imagen2 = cv2.imread('imagenprueba.jpg')

imagen3 = imagen1[1:568, 1:1001, :]

# fusion de imagenes
# toma el 0.7 de la primer imagen y el 0.3 de la segunda imagen
imagen_resultado = cv2.addWeighted(imagen3, 0.7, imagen2, 0.3, 0)
cv2.imshow('imagen_resultado', imagen_resultado)
cv2.imwrite('imagen_resultado.png', imagen_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
