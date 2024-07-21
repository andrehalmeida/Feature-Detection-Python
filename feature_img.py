import cv2 as cv
import matplotlib.pyplot as plt

pic1 = cv.imread('golf-gti.jpg', cv.IMREAD_ANYDEPTH) #Generates the first image
pic2 = cv.imread('hubcap.jpg', cv.IMREAD_ANYDEPTH) #Generates second image

orb = cv.ORB_create() #Creats an ORB object

kp1, des1 = orb.detectAndCompute(pic1, None)
kp2, des2 = orb.detectAndCompute(pic2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key= lambda x: x.distance)
pic3 = cv.drawMatches(pic1, kp1, pic2, kp2, matches[:11], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(pic3), plt.show()