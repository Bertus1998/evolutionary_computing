import random
import math

from deap import base
from deap import creator
from deap import tools
import numpy as np

from main import leeyao_func, schwefel_func, sphere_func, ackley_func, f2_func, f5_func


creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class GenericAlgorithm:
    def __init__(self, population: int, dimensions: int,  x_from: float, x_to: float, eval_func):
        self.eval_func = eval_func
        self.toolbox = base.Toolbox()
        # Attribute generator
        self.toolbox.register("attr_float", random.uniform, x_from, x_to)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                         self.toolbox.attr_float, n=dimensions)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.generic_eval_func)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.population = self.toolbox.population(n=population)
        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        self.CXPB, self.MUTPB = 0.5, 0.2

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in self.population]

    def generic_eval_func(self, x):
        return [self.eval_func(np.array(x))]

    def step(self):
        # Select the next generation individuals
        offspring = self.toolbox.select(self.population, len(self.population))
        # Clone the selected individuals
        offspring = list(map(self.toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.CXPB:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < self.MUTPB:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        self.population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in self.population]

        return min(fits)

    def run(self, mode, iterations=10000, epsilon=0.01):
        last_score = None
        if mode == 'iteration':
            for i in range(iterations):
                last_score = self.step()
                print(f'{i}: {last_score}')
            return last_score, iterations
        elif mode == 'epsilon':
            last_score = math.inf
            i = 0
            while last_score > epsilon and i < iterations:
                last_score = self.step()
                print(f'{i}: {last_score}')
                i += 1
            return last_score, i


if __name__ == '__main__':
    ga = GenericAlgorithm(300, 30, -100, 100, ackley_func)
    ga.run(mode='epsilon', epsilon=0.001)
