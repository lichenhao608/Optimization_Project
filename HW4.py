import sympy
import numpy as np 
from numpy.linalg import norm
import random
import time
import math
from numpy import * 
from gradient_test import gradient
from evaluate import *

def Rosenbrock(x, b=5, a=1):
    return (a-x[0]) ** 2 + b* (x[1]-x[0]**2) ** 2 + (a-x[1]) ** 2 + b * (x[2]-x[1]**2) ** 2 + (a-x[2]) ** 2 + b * (x[3]-x[2]**2) **2 + (a-x[3]) ** 2 + b * (x[4]-x[3]**2) ** 2 + (a-x[4]) ** 2 + b * (x[5]-x[4]**2) **2 + (a-x[5]) ** 2 + b * (x[6]-x[5]**2) **2 + (a-x[6]) ** 2 + b * (x[7]-x[6]**2)**2 + (a-x[7]) ** 2 + b * (x[8]-x[7]**2) ** 2 + (a-x[8]) ** 2 + b * (x[9]-x[8]**2) ** 2

def test_function(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 + x[5] ** 2 + (x[6]+1) ** 2 + (x[7]-5) ** 2 + (x[8]-5) ** 2 + (x[9]+6) ** 2

def constraint_function(x):
    return 3*x[0]**3 + x[1]**2 - x[2]**2 + x[3] ** 2 + 4 * x[4] + 5 * x[5] + 7 * x[6] + x[7] ** 2 - x[8]  + x[9] - 2

def gradient_descent(func, x0, p=0, c=0, rou=0, lam=0, alpha=0.001):

    x_final = x0
    f_final = func(x0)

    x = x0
    iter = 0
    precision = 1

    while iter < 10000 and norm(precision) > 1e-4:
        prev_x = x_final
        x_final = x_final - alpha * gradient(func, x_final)
        precision = abs(prev_x - x_final)
        iter += 1
   
    return x_final

def penalty_function(f, h, x, rou,lam):
    return f(x) + rou/2 * sum((h(x))**2) - (lam * h(x))

def augmented_lagrange_method(func, h, x, k_max):

    rou = 1
    y = 2
    lam = zeros(len(x))
    precision = 1
    iter = 0

    while iter < k_max and norm(precision) > 1e-6:

        prev_x = x
        x = gradient_descent(func,x, penalty_function, h, rou, lam)
        rou *= y
        lam -= rou*h(x)
        precision = abs(x-prev_x)

        iter += 1

    f = func(x)

    return x, f, iter

if __name__ == "__main__":

    result = evaluate(Rosenbrock, generate_random_num(),constraint_function, 1000)
    result_test = evaluate(test_function, generate_random_num(), constraint_function , 1000)

    result_table = [result, result_test]

    for i in range(0,2):
        for j in range(0,4):
            if i == 0 and j == 0:
                print('Test Augmented Lagrange Method on Rosenbrock\nXmean')
            elif i == 1 and j == 0:
                print('Test Augmented Lagrange Method on Test Function\nXmean')
            elif j == 1:
                print('FMean')
            elif j == 2:
                print('Iter Count Mean')
            elif j == 3:
                print('Wall Clock Time Mean')
            print(result_table[i][j])
        print('\n')


