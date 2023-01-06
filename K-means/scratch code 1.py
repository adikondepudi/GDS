from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import PIL
import pandas as pd
import seaborn_image as isns

img = PIL.Image.open("./Apples.jpg")
imgArray = np.array(img)
# imgArray = np.array(imgArray, dtype=np.float64) / 255
# print("Dimensions:", imgArray.ndim)
plt.imshow(imgArray, origin="upper")
#The image has two dimensions for the number of pixels horizontal and vertically. The third dimension is for the RGB value of the pixel. 
#Each channel represents Red, Green, Blue respectively. 
plt.show()