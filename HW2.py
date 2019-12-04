import numpy as np
import methods
import evaluate
from numpy import linalg as LA
import random

def quad_10(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

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

if __name__ == "__main__":
    initial_point_list = generate_initial_point_list(-5.0, 5.0)
    q_result = evaluate.evaluate("Gradient Descent", gradient_descent_method, quad_10, initial_point_list)
    
    result_table = [q_result]
    evaluate.print_table(result_table, "Gradient Descent Method", ["quad_10(x)=x[0]**2+x[1]**2+x[2]**2+x[3]**2+x[4]**2+x[5]**2+x[6]**2+x[7]**2+x[8]**2+x[9]**2"])
