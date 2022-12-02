#Code to generate artificial shapes from a random shape
#Code written by Anurag Shukla for class assignment of
#Computational Data Science course of PGDBA 2022-24

#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read the base shape from the image
df = pd.read_csv('mean_shape_hand.csv')
base_shape = df.values.tolist()
x_base = []
y_base = []
for i in base_shape:
    x_base.append(i[1])
    y_base.append(i[2])
#Generate noisy data
X = []
for i in range(50):
    temp_x = []
    for j in range(len(x_base)):
        noise = np.random.normal(0,4)
        noise_x = x_base[j] + noise
        temp_x.append(int(noise_x))
    for j in range(len(y_base)):
        noise = np.random.normal(0, 4)
        noise_y = y_base[j] + noise
        temp_x.append(int(noise_y))
    X.append(temp_x)

#finding cavariandce matrix of data
X_a = np.array(X)
cov_mat_x = []
for i in range(X_a.shape[1]):
    ctemp_x = []
    for j in range(X_a.shape[1]):
        a = np.cov(X_a[:,i],X_a[:,j])[0][1]
        ctemp_x.append(a)
    cov_mat_x.append(ctemp_x)

#eigenvectors and pca for 99% of values
l_x, vec_x = np.linalg.eig(cov_mat_x)
check = 0
for i in range(len(l_x)):
    check = check + l_x[i]
    if check/sum(l_x) > 0.99:
        break
pca = []
for j in range(i):
    pca.append(vec_x[:,j])
pca_a = np.array(pca).transpose()

#generating shapes using different beta values
beta = [2]*pca_a.shape[1]
x_mean = list(x_base)
for i in y_base:
    x_mean.append(i)
x_new = x_mean + np.dot(pca_a,beta)

#output base shape and new shape
xp = []
yp = []
for i in range(x_new.shape[0]):
    if i < x_new.shape[0]/2:
        xp.append(x_new[i].real)
    else:
        yp.append(x_new[i].real)

#completing the curve
x_base.append(x_base[0])
y_base.append(y_base[0])
xp.append(xp[0])
yp.append(yp[0])

plt.figure()
plt.plot(xp,yp)
plt.title("Synthetic Shape")
plt.figure()
plt.plot(x_base,y_base)
plt.title("Base Shape")
plt.show()
