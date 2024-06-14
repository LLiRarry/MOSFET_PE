from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
# Problem classes
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoode.algorithms import DE, GDE3, NSDE, NSDER
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoode.survival import RankAndCrowding, ConstrRankAndCrowding

from pymoo.termination.default import DefaultSingleObjectiveTermination, DefaultMultiObjectiveTermination
import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

libraries_path = "C:/PySpice/examples/libraries"
spice_library = SpiceLibrary(libraries_path)
file_path = 'C:\\PySpice\\examples\\libraries\\mosfet\\nmosmodelcard.lib'
from collections import OrderedDict

ordered_dict = OrderedDict()
ordered_dict['toxe'] = 1.85e-09
ordered_dict['xj'] = 1.5e-7
ordered_dict['ndep'] = 1.7e17
ordered_dict['k1'] = 0.497
ordered_dict['vth0'] = 0.429
ordered_dict['dvt0'] = 2.2
ordered_dict['u0'] = 0.04861
ordered_dict['vsat'] = 124340
ordered_dict['cgso'] = 1.5e-10
ordered_dict['cgdo'] = 1.5e-10
keys_list = list(ordered_dict.keys())
range_list = [(value * 0.1, value * 1.9) if isinstance(value, (int, float)) else (None, None) for value in
              ordered_dict.values()]


def chage_lib_file(Vars, path):
    OrderedDict_temp = OrderedDict()
    for i in range(len(keys_list)):
        OrderedDict_temp[keys_list[i]] = Vars[i]

    with open(path, 'w') as file:
        file.write("*****************************************************************\n")
        file.write(".model  nmosmodelcard  nmos  level = 54 version = 4.0 \n")
        for param_name, param_value in OrderedDict_temp.items():
            file.write(f"+{param_name}={param_value}\n")


def inverse_min_max_scaling_single(scaled_value, original_min, original_max):
    original_value = (scaled_value - 0.01) / (0.99 - 0.01) * (original_max - original_min) + original_min
    return original_value


def nna_opt(objective_function, LB, UB, nvars, npop, max_it):
    # Assuming LB, UB, npop, nvars, and objective_function are defined earlier
    # Create X_LB and X_UB
    X_LB = np.tile(LB, (npop, 1))
    X_UB = np.tile(UB, (npop, 1))
    beta = 1
    x_pattern = np.zeros((npop, nvars))
    cost = np.zeros(npop)
    # Create random initial population
    for i in range(npop):
        x_pattern[i, :] = LB + (UB - LB) * np.random.rand(1, nvars)
        cost[i] = objective_function(x_pattern[i, :])
    COST, index = np.min(cost), np.argmin(cost)
    # Create random initial weights with the constraint of the summation of each column equal to 1
    ww = np.ones(npop) * 0.5
    w = np.diag(ww)
    for i in range(npop):
        t = np.random.rand(1, npop - 1) * 0.5
        t = (t / np.sum(t)) * 0.5
        # w[:, i][w[:, i] == 0] = t
        w[w[:, i] == 0, i] = t
    #########################################
    XTarget = x_pattern[index, :]  # Best obtained solution
    Target = COST  # Best obtained objective function value
    wtarget = w[:, index]  # Best obtained weight (weight target)
    #########################################
    FMIN = np.zeros(max_it)
    for ii in range(1, max_it + 1):
        # Creating new solutions
        x_new = np.dot(w, x_pattern)
        x_pattern = x_new + x_pattern
        # Updating the weights
        for i in range(npop):
            w[:, i] = np.abs(w[:, i] + ((wtarget - w[:, i]) * 2 * np.random.rand(npop)))
        for i in range(npop):
            w[:, i] = w[:, i] / np.sum(w[:, i])  # Summation of each column = 1
        # Create new input solutions
        for i in range(npop):
            if np.random.rand() < beta:
                # Bias for input solutions
                N_Rotate = int(np.ceil(beta * nvars))
                xx = LB + (UB - LB) * np.random.rand(1, nvars)
                rotate_position = np.random.permutation(nvars)[:N_Rotate]
                for m in rotate_position:
                    x_pattern[i, m] = xx[0, m]
                # Bias for weights
                N_wRotate = int(np.ceil(beta * npop))
                w_new = np.random.rand(N_wRotate, npop)
                rotate_position_w = np.random.permutation(npop)[:N_wRotate]
                for j in range(N_wRotate):
                    w[rotate_position_w[j], :] = w_new[j, :]
                for iii in range(npop):
                    w[:, iii] = w[:, iii] / np.sum(w[:, iii])  # Summation of each column = 1
            else:
                # Transfer Function Operator
                x_pattern[i, :] = x_pattern[i, :] + (XTarget - x_pattern[i, :]) * 2 * np.random.rand(1, nvars)

        # Bias Reduction
        beta = beta * 0.99

        if beta < 0.01:
            beta = 0.05
        # beta = 1 - ((1 / max_it) * ii)  # An alternative way of reducing the value of beta
        # Check the side constraints
        x_pattern = np.maximum(x_pattern, X_LB)
        x_pattern = np.minimum(x_pattern, X_UB)

        # Calculating objective function values
        cost = np.array([objective_function(x) for x in x_pattern])

        # Selection
        FF, Index = np.min(cost), np.argmin(cost)

        if FF < Target:
            Target = FF
            XTarget = x_pattern[Index, :]
            wtarget = w[:, Index]
        else:
            _, Indexx = np.max(cost), np.argmax(cost)
            x_pattern[Indexx, :] = XTarget
            w[:, Indexx] = wtarget

        # Display
        print(f'Iteration: {ii}   Objective= {Target}   beta= {beta}')
        FMIN[ii - 1] = Target
    plt.plot(np.log(FMIN/10))
    np.save('Data2/NNA.npy', np.log(FMIN))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Objective Function Value')
    plt.show()
    Xoptimum = XTarget
    Objective = objective_function(Xoptimum)  # Best obtained optimum solution
    NFEs = npop * max_it
    return Xoptimum, Objective, NFEs


def RMSE(Vars):
    for i in range(10):
        Vars[i] = inverse_min_max_scaling_single(Vars[i], range_list[i][0], range_list[i][1])
    chage_lib_file(Vars, file_path)
    Vdd = 1.5
    VGS = []
    for i in range(6, 17, 2):
        VGS.append((i / 10))
    sim_re = []
    for i in VGS:
        circuit = Circuit('NMOS Transistor')
        circuit.include(spice_library['nmosmodelcard'])
        Vgate = circuit.V('gate', 'gatenode', circuit.gnd, i @ u_V)
        Vdrain = circuit.V('drain', 'vdd', circuit.gnd, 0 @ u_V)
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
        circuit.MOSFET(1, 'vdd', 'gatenode', circuit.gnd, circuit.gnd, model='nmosmodelcard')
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vdrain=slice(0, Vdd, .1))
        Sim_Vds = {}
        Sim_Vds['Vds'] = np.array(analysis['vdd']).tolist()
        Sim_Vds['Id'] = np.array(u_mA(-analysis.Vdrain)).tolist()
        Sim_Vds['Vgs'] = i
        sim_re.append(Sim_Vds)
    Id = []
    for i in range(len(sim_re)):
        Id.append(sim_re[i]['Id'])
    Id = np.array(Id).flatten()
    GroundTruth = np.load('../Data/Id.npy')
    GroundTruth = GroundTruth.flatten()
    mse = np.mean((Id - GroundTruth) ** 2)
    rmse = np.sqrt(mse)*10

    return rmse


LB, UB, nvars, npop, max_it = 0.01, 0.99, 10, 100, 300
Xoptimum, Objective, NFEs = nna_opt(RMSE, LB, UB, nvars, npop, max_it)
OrderedDict_temp = OrderedDict()
for i in range(10):
    OrderedDict_temp[keys_list[i]] = inverse_min_max_scaling_single(Xoptimum[i], range_list[i][0], range_list[i][1])
print(OrderedDict_temp)
