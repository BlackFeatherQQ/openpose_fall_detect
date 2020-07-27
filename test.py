import numpy
import cv2


I=numpy.zeros((416,416),dtype=numpy.uint8)

I=cv2.cvtColor(I,cv2.COLOR_GRAY2BGR)

cv2.imshow('test',I)
cv2.waitKey(0)