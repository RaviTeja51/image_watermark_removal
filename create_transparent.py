import numpy as np
import cv2

img_height = int(input("Enter the image height: "))
img_width = int(input("Enter the image width: "))
n = int(input("Enter the watermark_no: "))
transparent = np.zeros((img_height, img_width,4), dtype=np.uint8)

cv2.imwrite(f"transparent_img{n}.jpeg",transparent)
