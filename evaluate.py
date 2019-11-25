import sympy
import numpy as np 
from numpy.linalg import norm
import random
import time
import math
from numpy import * 
from gradient_test import gradient
from HW4 import augmented_lagrange_method

def random_sign():
    sign = '+'
    if random.randint(0,2) < 0.5:
        sign = '-'
    return sign 

def generate_random_num():
    random_num = []
    for i in range(0,50):
        temp = []
        for j in range(0,10):  
            if random_sign() == '+':
                y = 2 * random.random()
            else:
                y = -2 * random.random()
            x = np.exp(y)
            temp.append(x)
        random_num.append(temp)
    return random_num

def std(func, mean):
    sum = 0
    for i in func:
        temp = (i-mean) ** 2
        sum += temp
    std_mean = math.sqrt(sum/49)
    return std_mean

def std_vec(func, mean):
    
    sum = np.array([0,0,0,0,0,0,0,0,0,0])
    for i in func:
        temp = (i-mean) ** 2
        sum = np.add(temp, sum)
    std_mean = np.sqrt(sum/49)
    return std_mean

def evaluate(func, random_num, h, k_max):

    result_array = []

    x_total = [0,0,0,0,0,0,0,0,0,0]
    x_array = []
    f_total = 0
    f_array = []
    iter_total = 0
    iter_array = []
    time_total = 0
    time_array = []
   
    for i in random_num:
        
        t0 = time.time()
        x_final, f_final, iter = augmented_lagrange_method(func,h, i, k_max)
        process_time = time.time() - t0
        time_total += process_time
        x_array.append(x_final)
        x_total += x_final
        f_total += f_final
        f_array.append(f_final)
        iter_total += iter
        iter_array.append(iter)
        time_array.append(process_time)
            
    
    x_avg = x_total/50
    f_avg = f_total/50
    iter_avg = iter_total/50
    time_avg = time_total/50

    std_x = std_vec(x_array, x_avg)
    std_f = std(f_array, f_avg)
    std_time = std(time_array, time_avg)
    std_iter = std(iter_array, iter_avg)

    x = str(x_avg) + " \nplus minus\n " + str(std_x)
    f = str(f_avg) + " plus minus " + str(std_f)
    time_str = str(time_avg) + " plus minus " + str(std_time)
    iter_str = str(iter_avg) + " plus minus " + str(std_iter)

    result_array.append(x)
    result_array.append(f)
    result_array.append(iter_str)
    result_array.append(time_str)

    return result_array
