import numpy as np
import methods
import evaluate
from numpy import linalg as LA
import random
from numpy.linalg import norm

def quad_10(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

def Rosenbrock(x, b=5, a=1):
    return (a-x[0]) ** 2 + b* (x[1]-x[0]**2) ** 2 + (a-x[1]) ** 2 + b * (x[2]-x[1]**2) ** 2 + (a-x[2]) ** 2 + b * (x[3]-x[2]**2) **2 + (a-x[3]) ** 2 + b * (x[4]-x[3]**2) ** 2 + (a-x[4]) ** 2 + b * (x[5]-x[4]**2) **2 + (a-x[5]) ** 2 + b * (x[6]-x[5]**2) **2 + (a-x[6]) ** 2 + b * (x[7]-x[6]**2)**2 + (a-x[7]) ** 2 + b * (x[8]-x[7]**2) ** 2 + (a-x[8]) ** 2 + b * (x[9]-x[8]**2) ** 2

def generate_initial_point_list(lowerbound, upperbound, runs = 50, number_of_variables = 10):
    initial_point_list = []
    for r in range(0, runs):
        x_variables = []
        for n in range(0, number_of_variables):
            x = random.uniform(lowerbound, upperbound)
            x_variables.append(x)
        initial_point_list.append(x_variables)
    return initial_point_list

def gradient_descent_method(func, initial_point, initial_step_size):
    x_old = np.array(initial_point)
    y_old = func(initial_point)
    func_eval_count = 1
    threshold = 10**(-3)
    for n in range(0, 50000):
        direction = methods.gradient(func, x_old.tolist())
        x_new = x_old-initial_step_size*direction
        y_new = func(x_new.tolist())
        func_eval_count += 1
        if LA.norm(-direction) < threshold:
            return (x_new, y_new, func_eval_count)
        if y_new < y_old:
            x_old = x_new
            y_old = y_new
        else:
            initial_step_size *= 0.1
    return (x_old, y_old, func_eval_count)

def backtracking_line_search(f, x, d, alpha=0.001, p=0.5, beta=1e-4):
    y = f(x)
    g = methods.gradient(f, x)

    while f(x + alpha * d) > y + beta *  alpha * (np.dot(g,d)):
        alpha = alpha * 0.5

    return x + alpha * d

def conjugate_gradient_descent_method(func, x0, x_tol = 0.0005, f_tol = 0.01):
    x_final = x0
    f_final = func(x0)
    CG_iterations = 1

    g = methods.gradient(func, x0)
    d = -1 * g

    precision = 1

    while CG_iterations < 10000 and norm(precision) > x_tol:
        cur_x = x_final
        gprime = methods.gradient(func, x_final)
        beta = max(0, np.dot(gprime, gprime - g)/(np.dot(g,g)))
        dprime = -1 * gprime + beta * d
        x_final = backtracking_line_search(func,x_final,dprime)
        d = dprime
        g = gprime
        CG_iterations += 1
        precision = abs(x_final - cur_x)
      
    f_final = func(x_final)

    return x_final, f_final, CG_iterations

if __name__ == "__main__":
    initial_point_list = generate_initial_point_list(-5.0, 5.0)
    qg_result = evaluate.evaluate("Gradient Descent", gradient_descent_method, quad_10, initial_point_list)
    rg_result = evaluate.evaluate("Gradient Descent", gradient_descent_method, Rosenbrock, initial_point_list)
    qcg_result = evaluate.evaluate("Conjugate Gradient Descent", conjugate_gradient_descent_method, quad_10, initial_point_list)
    rcg_result = evaluate.evaluate("Conjugate Gradient Descent", conjugate_gradient_descent_method, Rosenbrock, initial_point_list)
    
    result_table = [qg_result, rg_result]
    evaluate.print_table(result_table, "Gradient Descent Method", ["quad_10(x)=x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2+x[5]**2+x[6]**2+x[7]**2+x[8]**2+x[9]**2", "Rosenbrock(x,b=5,a=1)=(a-x[0])**2+b*(x[1]-x[0]**2)**2+(a-x[1])**2+b*(x[2]-x[1]**2)**2+(a-x[2])**2+b*(x[3]-x[2]**2)**2+(a-x[3])**2+b*(x[4]-x[3]**2)**2+(a-x[4])**2+b*(x[5]-x[4]**2)**2+(a-x[5])**2+b*(x[6]-x[5]**2)**2+(a-x[6])**2+b*(x[7]-x[6]**2)**2+(a-x[7])**2+b*(x[8]-x[7]**2)**2+(a-x[8])**2+b*(x[9]-x[8]**2)**2"])

    result_table = [qcg_result, rcg_result]
    evaluate.print_table(result_table, "Conjugate Gradient Descent Method", ["quad_10(x)=x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2+x[5]**2+x[6]**2+x[7]**2+x[8]**2+x[9]**2", "Rosenbrock(x,b=5,a=1)=(a-x[0])**2+b*(x[1]-x[0]**2)**2+(a-x[1])**2+b*(x[2]-x[1]**2)**2+(a-x[2])**2+b*(x[3]-x[2]**2)**2+(a-x[3])**2+b*(x[4]-x[3]**2)**2+(a-x[4])**2+b*(x[5]-x[4]**2)**2+(a-x[5])**2+b*(x[6]-x[5]**2)**2+(a-x[6])**2+b*(x[7]-x[6]**2)**2+(a-x[7])**2+b*(x[8]-x[7]**2)**2+(a-x[8])**2+b*(x[9]-x[8]**2)**2"])