import cv2 as cv
import numpy
import sys

camara = cv.VideoCapture(0)

while(True):
    _, imagen_marco = camara.read()
    cv.imshow("Camara", imagen_marco)

    grises = cv.cvtColor(imagen_marco,cv.COLOR_RGB2GRAY)
    cv.imshow("Camara Gris", grises)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv.destroyAllWindows()