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


def project_to_feasible_region(x, lower_bound, upper_bound):
    # 将x中超出上下界的值投影回可行域内
    projected_x = np.minimum(np.maximum(x, lower_bound), upper_bound)
    return projected_x


def EO(Particles_no, Max_iter, lb, ub, dim, fobj, Run_no):
    Convergence_curve = np.zeros(Max_iter)
    Ave = np.zeros(Run_no)
    Sd = np.zeros(Run_no)

    for irun in range(1, Run_no + 1):
        Ceq1 = np.zeros(dim)
        Ceq1_fit = float('inf')
        Ceq2 = np.zeros(dim)
        Ceq2_fit = float('inf')
        Ceq3 = np.zeros(dim)
        Ceq3_fit = float('inf')
        Ceq4 = np.zeros(dim)
        Ceq4_fit = float('inf')

        C = initialization(Particles_no, dim, ub, lb)

        Iter = 0
        V = 1

        a1 = 1
        a2 = 1
        GP = 0.5

        while Iter < Max_iter:
            print('Iter', Iter)
            print(Ceq1_fit,"Ceq1_fit")
            fitness = np.zeros(Particles_no)

            for i in range(len(C)):
                Flag4ub = C[i, :] > ub
                Flag4lb = C[i, :] < lb
                C[i, :] = (C[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

                fitness[i] = fobj(C[i, :])

                if fitness[i] < Ceq1_fit:
                    Ceq1_fit = fitness[i]
                    Ceq1 = np.copy(C[i, :])
                elif Ceq1_fit < fitness[i] < Ceq2_fit:
                    Ceq2_fit = fitness[i]
                    Ceq2 = np.copy(C[i, :])
                elif Ceq2_fit < fitness[i] < Ceq3_fit:
                    Ceq3_fit = fitness[i]
                    Ceq3 = np.copy(C[i, :])
                elif Ceq3_fit < fitness[i] < Ceq4_fit:
                    Ceq4_fit = fitness[i]
                    Ceq4 = np.copy(C[i, :])

            if Iter == 0:
                fit_old = np.copy(fitness)
                C_old = np.copy(C)

            for i in range(Particles_no):
                if fit_old[i] < fitness[i]:
                    fitness[i] = fit_old[i]
                    C[i, :] = np.copy(C_old[i, :])

            C_old = np.copy(C)
            fit_old = np.copy(fitness)

            Ceq_ave = (Ceq1 + Ceq2 + Ceq3 + Ceq4) / 4
            C_pool = np.vstack([Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave])

            t = (1 - Iter / Max_iter) ** (a2 * Iter / Max_iter)

            for i in range(Particles_no):
                lambda_val = np.random.rand(dim)
                r = np.random.rand(dim)
                Ceq = C_pool[np.random.randint(0, C_pool.shape[0]), :]
                F = a1 * np.sign(r - 0.5) * (np.exp(-lambda_val * t) - 1)
                r1, r2 = np.random.rand(), np.random.rand()
                GCP = 0.5 * r1 * np.ones(dim) * (r2 >= GP)
                G0 = GCP * (Ceq - lambda_val * C[i, :])
                G = G0 * F
                C[i, :] = Ceq + (C[i, :] - Ceq) * F + (G / lambda_val * V) * (1 - F)

            Iter += 1
            Convergence_curve[Iter - 1] = Ceq1_fit

        Ave[irun - 1] = np.mean(Convergence_curve)
        Sd[irun - 1] = np.std(Convergence_curve)

        print('Run no:', irun)
        print('The best solution obtained by EO is:', Ceq1)
        print('The best optimal value of the objective function found by EO is:', Ceq1_fit)
        print('--------------------------------------')

    return Convergence_curve, Ave, Sd, Ceq1


def initialization(Particles_no, dim, ub, lb):
    return np.random.rand(Particles_no, dim) * (ub - lb) + lb


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
    rmse = np.sqrt(mse)

    return rmse


bounds = [(0.01, 0.99)] * 10

initial_guess = np.random.uniform(bounds[0][0], bounds[0][1], len(bounds))

# # Set EO parameters
Particles_no = 500
Max_iter = 300
Run_no = 1

# Run EO optimizer
Convergence_curve, Ave, Sd, optimized_params = EO(Particles_no, Max_iter, bounds[0][0], bounds[0][1],
                                                  len(initial_guess), RMSE, Run_no)

plt.plot(np.log(Convergence_curve))
np.save('Data2/EO2.npy', np.log(Convergence_curve))
plt.title("Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.show()
