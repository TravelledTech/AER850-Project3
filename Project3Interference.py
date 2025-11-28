from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train7/weights/best.pt")

img1 = cv2.imread("data/evaluation/ardmega.jpg")
img2 = cv2.imread("data/evaluation/arduno.jpg")
img3 = cv2.imread("data/evaluation/rasppi.jpg")

res1 = model.predict(img1)
res2 = model.predict(img2)
res3 = model.predict(img3)

img_1 = res1[0].plot(font_size=80, pil=True)
img_2 = res2[0].plot(font_size=20, pil=True)
img_3 = res3[0].plot(font_size=40, pil=True)

img_1.show()
img_2.show()
img_3.show()

img_1.save("img1.jpg")
img_2.save("img2.jpg")
img_3.save("img3.jpg")