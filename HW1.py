import evaluate
import math
import random

def f(x, a=2):
    return 1/2*((x-a)**2)

def g(x, a=2):
    return 1/4*(x**4)-a*x

def h(x, a=2):
    return math.exp(x)+math.exp(-x)-a*x

def golden_section_search_method(func, initial_point, initial_step_size):
    p = 1.618-1
    d = initial_point
    yd = func(d)
    func_eval_count = 1
    while True:
        c = d+initial_step_size
        yc = func(c)
        func_eval_count += 1
        if abs(yc-yd) <= 10**(-10):
            return (d, yd, func_eval_count)
        if yc <= yd:
            d, yd = c, yc
        else:
            direction = 1
            if evaluate.random_sign() == '-':
                direction = -1
            initial_step_size = initial_step_size*direction*p

def generate_initial_point_list(lowerbound, upperbound):
    initial_point_list = []
    for n in range(0, 100):
        y = random.uniform(lowerbound, upperbound)
        x = math.exp(y)
        direction = 1
        if evaluate.random_sign() == '-':
            direction = -1
        x = x*direction
        initial_point_list.append(x)
    return initial_point_list

if __name__ == "__main__":
    com_initial_point_list = generate_initial_point_list(-10.0, 10.0)
    h_initial_point_list = generate_initial_point_list(-2.0, 2.0)
    f_result = evaluate.evaluate("Golden Section Search", golden_section_search_method, f, com_initial_point_list)
    g_result = evaluate.evaluate("Golden Section Search", golden_section_search_method, g, com_initial_point_list)
    h_result = evaluate.evaluate("Golden Section Search", golden_section_search_method, h, h_initial_point_list)

    result_table = [f_result, g_result, h_result]
    evaluate.print_table(result_table, "Golden Section Search Method", ['f(x) = 1/2*((x-a)**2', 'g(x)=1/4*(x**4)-a*x', 'h(x)=e**x+e**(-x)-a*x'])
