import sympy
import numpy as np
from numpy.linalg import norm
import random
import time
import math
from numpy import *
from gradient_test import gradient


def random_sign():
    sign = '+'
    if random.randint(0, 2) < 0.5:
        sign = '-'
    return sign


def generate_random_num(runs=50, number_of_variables=10):
    random_num = []
    for i in range(0, runs):
        temp = []
        for j in range(0, number_of_variables):
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

    sum = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in func:
        temp = (i-mean) ** 2
        sum = np.add(temp, sum)
    std_mean = np.sqrt(sum/49)
    return std_mean


def evaluate(method, optimizer, func, starting_points, extra_parameters=None, k_max=1000):

    result_array = []

    x_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if method == "Golden Section Search":
        x_total = 0

    x_array = []
    f_total = 0
    f_array = []
    iter_total = 0
    iter_array = []
    time_total = 0
    time_array = []

    for i in starting_points:

        t0 = time.time()
        x_final, f_final, iter = 0, 0, 0
        if method == "Augmented Lagrange":
            x_final, f_final, iter = optimizer(
                func, extra_parameters["h"], i, k_max)
        elif method == "Golden Section Search":
            x_final, f_final, iter = optimizer(func, i, 100)
        elif method == "Gradient Descent":
            x_final, f_final, iter = optimizer(func, i, 10**(-3))
        elif method == "Conjugate Gradient Descent":
            x_final, f_final, iter = optimizer(func, i)
        elif method == "Quasi_Newton_BFGS":
            x_final, f_final, iter = optimizer(func, i)
        process_time = time.time() - t0
        time_total += process_time
        x_array.append(x_final)
        x_total += x_final
        f_total += f_final
        f_array.append(f_final)
        iter_total += iter
        iter_array.append(iter)
        time_array.append(process_time)

    x_avg = x_total/len(starting_points)
    f_avg = f_total/len(starting_points)

    iter_avg = iter_total/len(starting_points)
    time_avg = time_total/len(starting_points)

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
    result_array.append(min(f_array))

    return result_array


def print_table(result_table, Method, function_list):
    for i in range(0, len(result_table)):
        for j in range(0, 5):
            if j == 0:
                print('Test ' + Method + ' on ' + function_list[i] + '\nXmean')
            elif j == 1:
                print('FMean')
            elif j == 2:
                print('Iter Count Mean')
            elif j == 3:
                print('Wall Clock Time Mean')
            elif j == 4:
                print("FMin")
            print(result_table[i][j])
        print('\n')
