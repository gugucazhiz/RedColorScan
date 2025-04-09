import cv2
import numpy as np

#                H S V
# COR AMARELA
#lower = np.array([15,110,20])
#upper = np.array([40,255,255])
# COR VERMELHA
lower = np.array([0,120,70])
upper = np.array([11,255,255])

#COmo o vermelho ta sendo detctado de maneira fraca provavelmente por conta da camera
#vou colocar outra mascara pra detectar a faixa mais rosada do vermelho
lower1 = np.array([160,120,70])
upper1 = np.array([180,255,255])

camera = cv2.VideoCapture(0)

while True:
    sucess, img = camera.read()

    imagem1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(imagem1, lower, upper)
    mask2 = cv2.inRange(imagem1, lower1,upper1)
    #agora irei juntar os 2 arrays de cores
    mascara = cv2.bitwise_or(mask1,mask2)

    #Aumentando o tamanho dos pixels pretos e brancos
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.dilate(mascara, kernel, iterations=1)

    #Criando o contorno
    contornos, hierarquia = cv2.findContours(mascara, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contornos) != 0:
        for contor1 in contornos:
            if cv2.contourArea(contor1) > 500:
                x, y, w, h = cv2.boundingRect(contor1)
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Mascara",mascara)
    cv2.imshow("Normal", img)

    cv2.waitKey(1)
