import math
import random
from deap import base, creator, tools
import matplotlib.pyplot as plt

def getDistance(ind1, ind2):
    sum = 0
    for i in range(len(ind1)):
        sum += pow(ind1[i] - ind2[i], 2)
    return math.sqrt(sum)

def findMostSimilarInd(population, ind):
    most_similar = 0
    best_distance = getDistance(population[0], ind)
    count = len(population)
    for i in range(1, count):
        cur_distance = getDistance(population[i], ind)
        if (cur_distance < best_distance):
            most_similar = i
            best_distance = cur_distance
    return population[most_similar]


random.seed()
def randouble(minimum, maximum):
    return random.randrange(minimum, maximum)+random.random()

def evalFunction(individual):
    x = individual[0]
    y = individual[1]
    return (math.pow(x, 2)+math.pow(y, 2))/4000.0-math.cos(x)*math.cos(y/math.sqrt(2))+1,


def delta(genindex, y, rand, maxgen, decspeed):
    return y*(1-rand**((1-genindex/float(maxgen))**decspeed))

def mutNonuniform(individual, genindex, low, up, indpb, maxgen, decspeed):
    for i in range(0, len(individual), 1):
        if random.random() >= indpb:
            rand = random.random()
            if rand >= 0.5 :
                individual[i] += delta(genindex, up-individual[i], rand, maxgen, decspeed)
            else :
                individual[i] -= delta(genindex, individual[i]-low, rand, maxgen, decspeed)
    return individual,

#GA parameter
IND_SIZE = 2
POP_SIZE = 20
NEVAL = 100000
#crossover parameter
CXPB = 1
CXETA = 1
CXLOW = -30.0
CXUP = 30.0
#mutation parameter
MUTPB = 0.01
MUTETA = 100
MUTLOW = -30
MUTUP = 30
INDPB = 1
#RTS parameter
MAT_POOL_SIZE = 2
REP_POOL_SIZE = 5

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attribute", randouble, -90.0, 90.0)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalFunction)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,eta = CXETA, low = CXLOW, up = CXUP)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=MUTETA, low = MUTLOW, up = MUTUP, indpb = INDPB)
toolbox.register("select", tools.selRandom)

population = toolbox.population(n=POP_SIZE)
# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit
avr = []
mn =[]
for g in range(0, NEVAL, 2):
    # Select the next generation individuals
    parents = toolbox.select(population, MAT_POOL_SIZE)

    # Clone the selected individuals
    offsprings = [toolbox.clone(ind) for ind in parents]

    # Apply crossover on the offspring
    for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offsprings:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    for ind in offsprings:
        ind.fitness.values = toolbox.evaluate(ind)

    print(offsprings);
    # Restricted Tournament Selection
    for offspring in offsprings:
        replace_pool = toolbox.select(population, REP_POOL_SIZE)
        most_similar = findMostSimilarInd(replace_pool, offspring)
        if (offspring.fitness.values[0] < most_similar.fitness.values[0]):
            population.remove(most_similar)
            population.append(offspring)

    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5
    avr.append(mean)
    mn.append(min(fits))
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

# convergence process plotting
plt.plot(range(1, NEVAL + 1, 2), avr, 'ro', markersize=3)
plt.axis([1, NEVAL, 0, 2.5])
plt.xlabel('Evaluation #')
plt.ylabel('Average')
plt.show()
