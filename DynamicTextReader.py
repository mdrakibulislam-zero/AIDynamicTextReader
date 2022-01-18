import cv2
import copy
import numpy as np
from FaceMesh import FaceMeshDetector


cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

textList = ["Hi, I'm Md. Rakibul Islam",
            "Welcome to my digital playground!",
            "This is a dynamic text reader."]

sen = 25


def stackImages(_imgList, cols, scale):
    imgList = copy.deepcopy(_imgList)
    totalImages = len(imgList)
    rows = totalImages // cols if totalImages // cols * cols == totalImages else totalImages // cols + 1
    blankImages = cols * rows - totalImages
    width = imgList[0].shape[1]
    height = imgList[0].shape[0]
    imgBlank = np.zeros((height, width, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)

    # Resize The Images
    for i in range(cols * rows):
        imgList[i] = cv2.resize(imgList[i], (0, 0), None, scale, scale)
        if len(imgList[i].shape) == 2:
            imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_GRAY2BGR)

    # Put The Images In A Board
    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    return ver


def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255), colorR=(162, 155, 254),
                font=cv2.FONT_HERSHEY_PLAIN, offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)
    return img, [x1, y2, x2, y1]


while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3

        # Finding Distance
        f = 840
        d = (W * f) / w
        print(d)

        putTextRect(img, f'Depth: {int(d)}cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

        for i, text in enumerate(textList):
            singleHeight = 20 + int((int(d/sen)*sen)/4)
            scale = 0.4 + (int(d/sen)*sen)/75
            cv2.putText(imgText, text, (50, 50 + (i * singleHeight)), cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

    imgStacked = stackImages([img, imgText], 2, 1)
    cv2.imshow("Image", imgStacked)
    cv2.waitKey(1)
