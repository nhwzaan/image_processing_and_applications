import cv2
import matplotlib.pyplot as plt
import utils

path = utils.choose_img()
# print(path)

# img = cv2.imread(path)
img = plt.imread(path)
plt.imshow(img)

plt.show()