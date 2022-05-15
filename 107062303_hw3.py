import random

import numpy as np

# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
from HomeworkFramework import Function


class RS_optimizer(Function):  # need to inherit this class "Function"
    def __init__(self, target_func):
        super().__init__(target_func)  # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, FES):  # main part for your implementation

        while self.eval_times < FES:
            print('=====================FE=====================')
            print(self.eval_times)

            solution = np.random.uniform(np.full(self.dim, self.lower), np.full(self.dim, self.upper), self.dim)
            value = self.f.evaluate(func_num, solution)
            self.eval_times += 1

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break
            if float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: {}\n".format(self.get_optimal()[1]))


class DEOptimizer(Function):  # need to inherit this class "Function"
    def __init__(self, target_func, candidate_factor=6):
        super().__init__(target_func)  # must have this init to work normally

        self.lower = self.f.lower(target_func)
        self.upper = self.f.upper(target_func)
        self.dim = self.f.dimension(target_func)
        self.num_candidate = candidate_factor * self.dim

        self.target_func = target_func

        self.eval_times = 0
        self.optimal_value = float("inf")
        self.optimal_solution = np.empty(self.dim)

        self.theta = np.random.uniform(self.lower, self.upper, (self.dim, self.num_candidate))
        self.F = 0.25
        self.CR = 0.9

    def mutation(self):
        v = np.zeros_like(self.theta)
        for i in range(self.num_candidate):
            p, q, r = random.sample(range(self.num_candidate), 3)
            v[:, i] = self.theta[:, p] + self.F * (self.theta[:, q] - self.theta[:, r])
        return v

    def recombination(self, v):
        rand = np.random.rand(self.dim, self.num_candidate)
        u = np.where(rand <= self.CR, v, self.theta)
        i_rand = random.randrange(0, self.dim)
        u[i_rand] = v[i_rand]
        return np.clip(u, self.lower, self.upper)

    def selection(self, u):
        theta = self.theta.copy()
        for i in range(self.num_candidate):
            value = self.f.evaluate(self.target_func, self.theta[:, i])
            new_value = self.f.evaluate(self.target_func, u[:, i])

            self.eval_times += 1
            if 'ReachFunctionLimit' in (value, new_value):
                print('ReachFunctionLimit')
                break
            if value < self.optimal_value:
                self.optimal_solution[:] = self.theta[:, i]
                self.optimal_value = value

            theta[:, i] = u[:, i] if new_value < value else self.theta[:, i]
        return theta

    def optimization_loop(self):
        v = self.mutation()
        u = self.recombination(v)
        self.theta = self.selection(u)

    def get_optimal(self):
        return self.optimal_solution, self.optimal_value

    def run(self, fes):  # main part for your implementation

        while self.eval_times < fes:
            print('=====================FE=====================')
            print(self.eval_times)

            self.optimization_loop()

            print("optimal: {}\n".format(self.get_optimal()[1]))


if __name__ == '__main__':
    func_num = 1
    fes = 0
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
    while func_num < 5:
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        # you should implement your optimizer
        op = DEOptimizer(func_num)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
        func_num += 1
