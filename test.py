import numpy
import cv2
#
#
# I=numpy.zeros((416,416),dtype=numpy.uint8)
#
# I=cv2.cvtColor(I,cv2.COLOR_GRAY2BGR)
#
# cv2.imshow('test',I)
# cv2.waitKey(0)

import threading
import time
import glob
import os
import concurrent.futures
import multiprocessing



# def load_and_resize(image_filename):
#     img = cv2.imread(image_filename)
#
#     img = cv2.resize(img,(416,416))
#
#     print(image_filename)
#
# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         image_files = glob.glob("C:/Users/lieweiai/Desktop/human_pose/train/normal/*.jpg")
#
#         # print(image_files.shape)
#
#         executor.map(load_and_resize, image_files)
# import multiprocessing
# import time
# def func(msg,I):
#   for i in range(3):
#     print(msg)
#     time.sleep(1)
#   return "done " + msg
# if __name__ == "__main__":
#   pool = multiprocessing.Pool(processes=4)
#   result = []
#   for i in range(10):
#     msg = "hello %d" %(i)
#     result.append(pool.apply_async(func, (msg,  i)))
#   pool.close()
#   pool.join()
#   for res in result:
#     print(res.get())

