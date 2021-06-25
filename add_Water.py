import numpy as np
import cv2
import os


n = int(input("Enter number: "))
file = input("Enter file path: ")
overlay = cv2.imread(f"transparent_img{n}.jpg")

background =  cv2.imread(file)
shape = background.shape
overlay_reshape=cv2.resize(overlay,(shape[1],shape[0]))
alpha = 0.8
beta = 0.2
added_image = cv2.addWeighted(src1=background,src2=overlay_reshape,alpha=alpha,beta=beta,gamma=0)
cv2.imwrite(f"watermark_{file}_{alpha}.jpeg",added_image)
