import abc
import random

import numpy as np

# you must use python 3.6, 3.7, 3.8, 3.9 for sourcedefender
import sourcedefender
from HomeworkFramework import Function
from typing import Tuple, List


class RSOptimizer(Function):  # need to inherit this class "Function"
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

    def run(self, fes):  # main part for your implementation
        while self.eval_times < fes:
            print('=====================FE=====================')
            print(self.eval_times)

            solution = np.random.uniform(self.lower, self.upper, self.dim)
            value = self.f.evaluate(self.target_func, solution)
            self.eval_times += 1

            if value == "ReachFunctionLimit":
                print("ReachFunctionLimit")
                break
            if float(value) < self.optimal_value:
                self.optimal_solution[:] = solution
                self.optimal_value = float(value)

            print("optimal: {}\n".format(self.get_optimal()[1]))


class Strategy(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def mutation(theta: np.ndarray, idx: int, f: float) -> np.ndarray:
        """ Mutate the parameter theta then return the mutated one """

    @staticmethod
    @abc.abstractmethod
    def recombination(theta: np.ndarray, v: np.ndarray, cr: float) -> np.ndarray:
        """ Recombine original params `theta` and mutated params `v` """


class BinaryRecombine(Strategy):
    @staticmethod
    def recombination(theta: np.ndarray, v: np.ndarray, cr: float) -> np.ndarray:
        rand = np.random.rand(*theta.shape)
        u = np.where(rand <= cr, v, theta)
        i_rand = random.randrange(0, theta.shape[0])
        u[i_rand] = v[i_rand]
        return u


class Rand1Bin(BinaryRecombine):
    @staticmethod
    def mutation(theta: np.ndarray, idx: int, f: float) -> np.ndarray:
        p, q, r = random.sample(range(theta.shape[1]), 3)
        return theta[:, p] + f * (theta[:, q] - theta[:, r])


class Rand2Bin(BinaryRecombine):
    @staticmethod
    def mutation(theta: np.ndarray, idx: int, f: float) -> np.ndarray:
        p, q, r, s, t = random.sample(range(theta.shape[1]), 5)
        return theta[:, p] + f * (theta[:, q] - theta[:, r]) + f * (theta[:, s] - theta[:, t])


class CurrentToRand1(Strategy):
    @staticmethod
    def mutation(theta: np.ndarray, idx: int, f: float) -> np.ndarray:
        p, q, r = random.sample(range(theta.shape[1]), 3)
        rand = np.random.rand()
        return theta[:, idx] + rand * ((theta[:, p] - theta[:, idx]) + f * (theta[:, q] - theta[:, r]))

    @staticmethod
    def recombination(theta: np.ndarray, v: np.ndarray, cr: float) -> np.ndarray:
        return v


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
        self.strategy = Rand1Bin
        self.F = 0.25
        self.CR = 0.9

    def mutation(self):
        v = np.zeros_like(self.theta)
        for i in range(self.num_candidate):
            v[:, i] = self.strategy.mutation(self.theta, i, self.F)
        return v

    def recombination(self, v):
        u = self.strategy.recombination(self.theta, v, self.CR)
        return np.clip(u, self.lower, self.upper)

    def selection(self, u):
        theta = self.theta.copy()
        for i in range(self.num_candidate):
            value = self.f.evaluate(self.target_func, self.theta[:, i])
            new_value = self.f.evaluate(self.target_func, u[:, i])

            self.eval_times += 2
            if 'ReachFunctionLimit' in (value, new_value):
                print('ReachFunctionLimit')
                break
            if new_value < self.optimal_value:
                self.optimal_solution[:] = self.theta[:, i]
                self.optimal_value = new_value

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


class CoDEOptimizer(DEOptimizer):
    def __init__(self, target_func, candidate_factor=6):
        super().__init__(target_func, candidate_factor)

        self.strategy_pool = [Rand1Bin, Rand2Bin, CurrentToRand1]
        self.param_pool = [(1.0, 0.3), (1.0, 0.9), (0.4, 0.2), (0.25, 0.9)]

    def selection(self, us):
        theta = self.theta.copy()
        for i in range(self.num_candidate):
            value = self.f.evaluate(self.target_func, self.theta[:, i])
            new_values = [self.f.evaluate(self.target_func, u[:, i]) for u in us]

            self.eval_times += 1 + len(new_values)
            if 'ReachFunctionLimit' in (value, *new_values):
                print('ReachFunctionLimit')
                break

            min_arg = np.argmin(new_values)
            min_value = new_values[min_arg]
            if min_value < self.optimal_value:
                self.optimal_solution[:] = self.theta[:, i]
                self.optimal_value = min_value

            theta[:, i] = us[min_arg][:, i] if min_value < value else self.theta[:, i]
        return theta

    def optimization_loop(self):
        us = []
        for self.strategy in self.strategy_pool:
            self.F, self.CR = random.choice(self.param_pool)
            # print(f'{self.strategy} with {self.F}, {self.CR}')
            v = self.mutation()
            u = self.recombination(v)
            us.append(u)
        self.theta = self.selection(us)


if __name__ == '__main__':
    func_num = 1
    fes = 0
    # function1: 1000, function2: 1500, function3: 2000, function4: 2500
    for func_num in range(1, 5):
        if func_num == 1:
            fes = 1000
        elif func_num == 2:
            fes = 1500
        elif func_num == 3:
            fes = 2000
        else:
            fes = 2500

        # you should implement your optimizer
        op = CoDEOptimizer(func_num, 5)
        op.run(fes)

        best_input, best_value = op.get_optimal()
        print(best_input, best_value)

        # change the name of this file to your student_ID and it will output properlly
        with open("{}_function{}.txt".format(__file__.split('_')[0], func_num), 'w+') as f:
            for i in range(op.dim):
                f.write("{}\n".format(best_input[i]))
            f.write("{}\n".format(best_value))
