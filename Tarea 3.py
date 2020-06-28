# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:23:37 2020

@author: Roberto
"""

import numpy as np
from numpy import exp, linspace, random
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
from fitter import Fitter
from pylab import linspace, plot
from mpl_toolkits import mplot3d


"""
 xy.csv sin primera fila y columna
"""

xy1 = np.loadtxt(open("xy.csv", "r"),str, delimiter=",")
xy2 = np.delete(xy1, 0, 0)
xy3 = np.delete(xy2, 0, 1)
xy = xy3.astype(float)

"""
 xyp.csv sin primera fila
"""
xyp1 = np.loadtxt(open("xyp.csv", "r"),str, delimiter=",")
xyp3 = np.delete(xyp1, 0, 0)
xyp= xyp3.astype(float)


"""
xs y ys
"""
xs = np.arange(5,16)
ys = np.arange(5,26)

"""
Marginal de X y Y
"""
fX = np.sum(xy, axis =1)
fY = np.sum(xy, axis =0)

"""
Graficando Marginales
"""
f1 = plt.figure()
plt.plot(xs,fX)
plt.title("Marginal de X")
f2 = plt.figure()
plt.plot(ys,fY)
plt.title("Marginal de Y")

"""
Funcion Gaussiana
"""
def gauss(x,mu,sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x - mu)**2/(2*sigma**2))

"""
Mejor ajuste con base en la Gaussiana
"""

paramX, _ = curve_fit(gauss,xs,fX)
paramY, _ = curve_fit(gauss,ys,fY)
print("Los parametros de la distribucion marginal gaussiana de X son:",paramX)
print("Los parametros de la distribucion marginal gaussiana de Y son:",paramY)
muX = 9.90484381
sigmaX = 3.29944287
muY = 15.0794609
sigmaY = 6.02693776

"""
Graficas de mejores ajustes de las marginales 2D:
"""

### Ajuste Gaussiano de X con 100 puntos:
xscont = np.linspace(5,15,100)
f3 = plt.figure()
plt.plot(xscont,gauss(xscont,9.90484381,3.29944287))
plt.title("Ajuste Gaussiano para marginal de X")

### Ajuste Gaussiano de Y con 100 puntos:
yscont = np.linspace(5,25,100)
f4 = plt.figure()
plt.plot(yscont,gauss(yscont,15.0794609,6.02693776))
plt.title("Ajuste Gaussiano para marginal de Y")


"""
Funcion de densidad conjunta

(Aqui se tuvo que transponer uno de los ajustes marginales para que a la hora de multiplicarlos
se obtuviera una matriz que diera origen a una superficie)
"""

### Para X se hizo una matriz de una fila y varias columnas:
ajusX1 = gauss(xscont,9.90484381,3.29944287)
ajusX= np.array([ajusX1])
print("Las dimensiones de la matriz de ajuste de X son:", np.shape(ajusX))

### Para Y se hizo una matriz de una columna y varias filas:
ajusY1 = gauss(yscont,15.0794609,6.02693776)
ajusY2= np.array([ajusY1])
ajusY= ajusY2.transpose()
print("Las dimensiones de la matriz de ajuste de Y son:", np.shape(ajusY))

### Multiplicando las matrices anteriores para obtener la matriz de densidad conjunta:

ajusconj = ajusX*ajusY
print("Las dimensiones de la matriz de densidad conjunta son:", np.shape(ajusconj))



"""
Grafica de mejor ajuste conjunto 3D:
"""
zscont = np.linspace(0,1,100)

f5 = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(xscont, yscont)
ax.plot_wireframe(X,Y, ajusconj, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Densidad de probabilidad');
ax.set_title('Funcion de densidad conjunta')

f6 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, ajusconj, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Densidad de probabilidad');
ax.set_title('Funcion de densidad conjunta con colorcito')



"""
Correlacion
"""
multi1 = np.prod(xyp, axis = 1)
correlacion= np.sum(multi1)
print("La correlacion es:", correlacion)


"""
Covarianza
"""
### Restando las medias y elevando al cuadrado:

for i in range(231):
    xyp[i][0] -= muX
    xyp[i][1] -= muY

    
### Obteniendo la covarianza:

multi2 = np.prod(xyp, axis = 1)
covarianza= np.sum(multi2)
print("La covarianza es:", covarianza)


"""
Coeficiente de correlacion
"""
coef = covarianza/(sigmaX*sigmaY) 
print("El coeficiente de correlacion es:", coef)

print("Se puede concluir que hay practicamente una independencia de X y Y por muchas razones:  La correlacion E[XY] es casi igual a E[X]*E[Y], la covarianza es casi igual a 0 y, por medio del coeficiente de correlacion (casi igual a cero tambien) nos damos cuenta que si en la funcion de densidad conjunta (3D) se hiciesen cortes en el eje Z, los puntos que quedarian en dichos cortes no estarian ubicados colinealmente .")







































