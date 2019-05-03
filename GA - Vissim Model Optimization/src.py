import logger
import logging
import config
import pandas as pd
import datetime
import win32com.client as com
import sqlite3
import random
import numpy as np
import math
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import pickle

def decode(max, min, individual):
    """
    Decode binary to real number.

    :type max: int
    :param max: max range of parameter
    :type min: int
    :param min: min range of parameter
    :type individual: toolbox object
    :param individual: gene from GA run holding binary info, e.g. "0001011100"
    :return: numpy.float 64
    """
    max_f = float(max)
    min_f = float(min)

    n = float(len(individual))
    alpha = (max_f-min_f)/(2.0**n-1.0)
    beta = min_f
    A = individual
    B = []
    for i in range(1,int(n)+1):
        x = 2**(i-1)
        B.append(x)

    output = np.dot(np.dot(alpha, A), list(reversed(B))) + beta
    return output

def runSim(input):
    """
    Call Vissim application through com object, load Vissim inpx file
    and run simulation with input parameters.

    :type input: dictionary
    :param input: decoded parameters
    :return: pandas dataframe
    """
    cfg = config.run_config()
    path = 'C:\Users\\nli\PycharmProjects\GA sim\hwy_1_corridor(autorun).inpx'
    run = cfg['run']
    period = cfg['period']

    def genQuery(input):
        x = 'TravTm(' + str(input) + ',1,All)'
        return x

    Vissim = com.Dispatch('Vissim.Vissim')
    Vissim.LoadNet(path)
    db = Vissim.Net.DrivingBehaviors.ItemByKey(3)

    Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode",1)
    Vissim.SuspendUpdateGUI()
    Vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)
    Vissim.Simulation.SetAttValue('NumRuns', run)
    Vissim.Simulation.SetAttValue('SimPeriod', period)

    db.SetAttValue('W99cc0', input['cc0'])
    db.SetAttValue('W99cc1Distr', input['cc1'])
    db.SetAttValue('W99cc4', input['cc4'])
    db.SetAttValue('W99cc5', input['cc5'])

    Vissim.Simulation.RunContinuous()

    # append results from each simulation run to table
    result = []
    for i in range(1, run+1):
        r = Vissim.Net.VehicleTravelTimeMeasurements.GetMultipleAttributes(('Name', genQuery(i)))
        result.append(list(r))

    df = pd.DataFrame(np.array(result).reshape(run,2), columns=['Route','TravTm'])
    df['TravTm'] = df['TravTm'].apply(pd.to_numeric)

    return df

def mockSim(input):
    t = input['cc0'] + input['cc1'] + input['cc4'] + input['cc5']
    result = ['First to Hasting', t]
    df = pd.DataFrame(np.array(result).reshape(1,2), columns=['Route','TravTm'])
    df['TravTm'] = df['TravTm'].apply(pd.to_numeric)
    return df

def eval_SSE(individual):
    """
    Calculate SSE for travel time from each simulation iteration.

    :type individual: toolbox object
    :param individual: gene from GA run holding binary info, e.g. "0001011100"
    :return: a single element tuple
    """
    sqlite_file = 'vissim_db.sqlite'
    conn = sqlite3.connect(sqlite_file)
    file = 'TravelTime_Google.csv'
    benchmark = pd.read_csv(file)

    cfg = config.db_config()
    cc0 = decode(cfg['cc0_max'], cfg['cc0_min'], individual[:7])
    cc1 = int(decode(cfg['cc1_max'], cfg['cc1_min'], individual[7:9]))
    cc4 = decode(cfg['cc4_max'], cfg['cc4_min'], individual[9:12])
    cc5 = decode(cfg['cc5_max'], cfg['cc5_min'], individual[-3:])
    input = {'cc0':cc0, 'cc1':cc1, 'cc4':cc4, 'cc5':cc5}

    result = mockSim(input)
    x =  benchmark.loc[benchmark.Route == 10, 'Average of Travel_Time(min)'].iloc[0] * 60
    result['Diff'] = result['TravTm'] - x
    sse = sum(result['Diff']**2)
    SSE = (sse, )

    finalResult = pd.DataFrame({'Route':result.iloc[0,0], 'Diff':np.mean(result['Diff']),
                                'SSE':sse,'cc0':cc0, 'cc1':cc1,'cc4':cc4, 'cc5':cc5,
                                'Timestamp':datetime.datetime.now()}, index = [0])
    finalResult.to_sql('result', conn, if_exists='append', index=False)

    return SSE

def num_of_gene(max, min, incr):
    max_f = float(max)
    min_f = float(min)
    incr_f = float(incr)
    n = round(math.log((max_f-min_f)/incr_f+1,2),0)
    return int(n)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# calculate required length for individuals
cfg = config.db_config()
n = num_of_gene(cfg['cc0_max'], cfg['cc0_min'], cfg['cc0_incr']) + \
    num_of_gene(cfg['cc1_max'], cfg['cc1_min'], cfg['cc1_incr']) + \
    num_of_gene(cfg['cc4_max'], cfg['cc4_min'], cfg['cc4_incr']) + \
    num_of_gene(cfg['cc5_max'], cfg['cc5_min'], cfg['cc5_incr'])

# structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_SSE)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(checkpoint=None, FREQ=2, verbose=True):
    """
    This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param checkpoint: string, checkpoint file path
    :param FREQ: int, save result to checkpoint every n*FREQ generation
    :param verbose: boolean, whether or not to log the statistics
    :return: A list of varied individuals that are independent of their
             parents.
    """
    random.seed(318)

    # probability of crossover
    cxpb = 0.5
    # probability for mutation
    mutpb = 0.2
    # number of generations
    ngen = 2

    if checkpoint:
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)

        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        population = toolbox.population(n=70)
        start_gen = 0
        halloffame = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()

    # set initial configuration
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = mstats.compile(population) if mstats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen+ 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = mstats.compile(population) if mstats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print logbook.stream

        if gen % FREQ == 0:
            cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())

            with open("checkpoint_name.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    return population, logbook

if __name__ == "__main__":
    logger.createLogger()
    try:
        main()
    except Exception:
        logging.exception("KeyError")