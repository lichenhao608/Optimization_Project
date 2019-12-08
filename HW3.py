import numpy as np
import methods
import evaluate
import random
from numpy.linalg import norm

def quad_10(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

def Rosenbrock(x, b=5, a=1):
    return (a-x[0]) ** 2 + b* (x[1]-x[0]**2) ** 2 + (a-x[1]) ** 2 + b * (x[2]-x[1]**2) ** 2 + (a-x[2]) ** 2 + b * (x[3]-x[2]**2) **2 + (a-x[3]) ** 2 + b * (x[4]-x[3]**2) ** 2 + (a-x[4]) ** 2 + b * (x[5]-x[4]**2) **2 + (a-x[5]) ** 2 + b * (x[6]-x[5]**2) **2 + (a-x[6]) ** 2 + b * (x[7]-x[6]**2)**2 + (a-x[7]) ** 2 + b * (x[8]-x[7]**2) ** 2 + (a-x[8]) ** 2 + b * (x[9]-x[8]**2) ** 2

def get_identity_matrix(length):
    a = np.identity(length)
    return a

def generate_random_num():
    random_num = []
    for i in range(0,50):
        temp = []
        for j in range(0,10):  
            if evaluate.random_sign() == '+':
                y = 2 * random.randint(0,2)
            else:
                y = -2 * random.randint(0,2)
            x = np.exp(y)
            temp.append(x)
        random_num.append(temp)
    return random_num

def line_search(func, g, x ,s, p=0.5, beta=10):
    #obtained from github located at https://github.com/tazzben/EconScripts/blob/master/Optimization%20Scripts/Python/bfgs.py

    def find(alpha):
        if abs(alpha) < 1e-5:
            return 1
        return (func(x+alpha * s) - func(x))/alpha * np.dot(g,s)

    alpha = 1
 
    while find(alpha) >= 0.1:
        alpha *= 2

    while find(alpha) < 0.1:
        temp = alpha / (2.0 * (1-find(alpha)))
        alpha = max(1.0/beta * alpha, temp)
    
    return alpha

def Quasi_Newton_BFGS_method(f, x0, step = 0, precision = 1):
    #The Broyden-Fletcher-Goldfarb-Shanno Descent method
    #Acquired From K&W P93

    #initial setting
    m = len(x0)
    q = get_identity_matrix(m)
    g = methods.gradient(f, x0)
    x = x0

    while step < 10000 and norm(precision) > 1e-5:
        s = -np.dot(q, g)
        alpha = line_search(f, g, x, s)
        prev_x = x
        x = x + alpha * s

        precision = abs(prev_x - x) 
        g_old = g
        g = methods.gradient(f, x)

        y = (g - g_old)/alpha
        dot_sy = np.dot(s,y)
        if dot_sy > 0:
            z = np.dot(q,y)
            q += np.outer(s,s)*(np.dot(s,y) + np.dot(y, z))/dot_sy**2 - (np.outer(z,s)+ np.outer(s, z))/dot_sy

        step += 1

    f_final = f(x)

    return x, f_final, step

if __name__ == "__main__":
    q_result = evaluate.evaluate("Quasi_Newton_BFGS", Quasi_Newton_BFGS_method, quad_10, generate_random_num())
    r_result = evaluate.evaluate("Quasi_Newton_BFGS", Quasi_Newton_BFGS_method, Rosenbrock, generate_random_num())

    result_table = [q_result, r_result]

    evaluate.print_table(result_table, "Quasi Newton BFGS Descent Method", ["quad_10(x)=x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2+x[5]**2+x[6]**2+x[7]**2+x[8]**2+x[9]**2", "Rosenbrock(x,b=5,a=1)=(a-x[0])**2+b*(x[1]-x[0]**2)**2+(a-x[1])**2+b*(x[2]-x[1]**2)**2+(a-x[2])**2+b*(x[3]-x[2]**2)**2+(a-x[3])**2+b*(x[4]-x[3]**2)**2+(a-x[4])**2+b*(x[5]-x[4]**2)**2+(a-x[5])**2+b*(x[6]-x[5]**2)**2+(a-x[6])**2+b*(x[7]-x[6]**2)**2+(a-x[7])**2+b*(x[8]-x[7]**2)**2+(a-x[8])**2+b*(x[9]-x[8]**2)**2"])
