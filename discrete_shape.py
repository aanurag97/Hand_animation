#Code to read image and generate a base shape using split & merge
#algorithm and export data into a CSV file. Code written by
#Anurag Shukla for class assignment of Computational Data Science
#course of PGDBA 2022-24

#import necessary files
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

#read image and define co-ordinate plane
img = cv2.imread("hand.png", 0)
origin = int(img.shape[0]/2), int(img.shape[1]/2)
'''
#Count total number of Data Points
count = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] < 125:
            count = count + 1

print(count)
'''
#find random initial point
flag = 0
for i in range(img.shape[0]):
    if flag == 1:
        break
    for j in range(img.shape[1]):
        if img[i, j] == 0:
            point = [i,j]
            flag = 1
            break

#8 neighbour search over the image
def knn(img, point):
    main_shape = []
    x,y = point
    while 1:
        main_shape.append([x, y])
        img[x,y] = 125
        if img[x,y+1] == 0:
            y = y + 1
        elif img[x+1,y+1] == 0:
            x, y = x + 1, y + 1
        elif img[x+1,y] == 0:
            x = x + 1
        elif img[x+1,y-1] == 0:
            x, y = x + 1, y - 1
        elif img[x,y-1] == 0:
            y=y-1
        elif img[x-1,y-1] == 0:
            x, y = x - 1, y - 1
        elif img[x-1,y] == 0:
            x = x-1
        elif img[x-1,y+1] == 0:
            x,y = x-1,y+1
        else:
          break
    return main_shape

#base image
main_shape = knn(img, point)
x_base = []
y_base = []
for i in main_shape:
    x_base.append(i[0])
    y_base.append(i[1])
plt.figure()
plt.plot(x_base,y_base)
plt.title("Base Shape")

'''
#split & merge using distance method
def split_merge_d(x,y,epsilon):
    critical_points = []
    i = 0
    while i < x.shape[0]:
        x1, y1 = x[i], y[i]
        for j in range(i + 1, x.shape[0]):
            x2, y2 = x[j], y[j]
            dist = math.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
            if dist > epsilon:
                critical_points.append([x1, y1])
                i = j
                break
        i = i + 1
    return critical_points
'''
#split & merge using slope method
def split_merge_s(x, y, slope, origin):
    critical_points = []
    i = 0
    while i < x.shape[0]:
        x1,y1 = x[i],y[i]
        for j in range(i+1, x.shape[0]):
            x2,y2 = x[j],y[j]
            m = math.atan((y2-origin[1])/(x2-origin[0])) - math.atan((y1-origin[1])/(x1-origin[0]))
            dist = math.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
            if (abs(m) > slope and dist > 50) or dist > 100:
                critical_points.append([x1,y1])
                i = j
                break
        i = i + 1
    return critical_points

#parameters for critical poitns
#use split_merge_d for distance algorithm along
#with epsilon_d. use split_merge_s for slope algorithm
#along with epsilon_s
#epsilon_d = 50
epsilon_s = math.pi/15
x_array = np.array(x_base)
y_array = np.array(y_base)
#test = split_merge_d(x_array,y_array, epsilon_d)
test = split_merge_s(x_array,y_array, epsilon_s, origin)
x_coor = []
y_coor = []
for i in test:
    x_coor.append(i[0])
    y_coor.append(i[1])

#plotting reduced shape
plt.figure()
plt.scatter(x_coor,y_coor)
plt.title("Reduced Shape")
plt.show()

#export as csv file
dict = {'x': x_coor, 'y': y_coor}
df = pd.DataFrame(dict)
df.to_csv('mean_shape_hand.csv')

