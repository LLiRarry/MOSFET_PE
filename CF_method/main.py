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

def chage_lib_file(Vars, path, index):
    OrderedDict_temp = OrderedDict()
    for i in range(len(keys_list)):
        OrderedDict_temp[keys_list[i]] = Vars[i][index]

    with open(path, 'w') as file:
        file.write("*****************************************************************\n")
        file.write(".model  nmosmodelcard  nmos  level = 54 version = 4.0 \n")
        for param_name, param_value in OrderedDict_temp.items():
            file.write(f"+{param_name}={param_value}\n")


def MSE_RMSE(Vars, GroundTruth, popsize):
    RMSEs = []
    for pop in range(popsize):
        # Context from Function C:/Users/czh/PycharmProjects/PPO_PE/CF_method/main.py:chage_lib_file
        chage_lib_file(Vars, file_path, pop)
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
        GroundTruth = GroundTruth.flatten()
        mse = np.mean((Id - GroundTruth) ** 2)
        rmse = np.sqrt(mse)
        RMSEs.append(rmse)
    RMSEs = np.array(RMSEs).reshape(popsize, 1)

    return RMSEs


def inverse_min_max_scaling_single(scaled_value, original_min, original_max):
    original_value = (scaled_value - 0.01) / (0.99 - 0.01) * (original_max - original_min) + original_min
    return original_value

convert=[]
class TwoObjExample(Problem):

    def __init__(self):
        xl = np.full(10, 0.01)
        xu = np.full(10, 0.99)
        super().__init__(n_var=10, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)
        self.ground_data = np.load('../Data/Id.npy')

    def _evaluate(self, x, out, *args, **kwargs):
        Vars = []
        for i in range(10):
            Vars.append(inverse_min_max_scaling_single(x[:, i], range_list[i][0], range_list[i][1]))
        F= MSE_RMSE(Vars, self.ground_data, popsize)  # list->scalar1, list->scalar2
        convert.append(np.min(F))
        # if len(convert)>300:
        #     np.save('Data2/PSO.npy', np.log(convert))
        out["F"] = F


popsize = 20
gde3 = GDE3(pop_size=popsize, variant="DE/rand/1/bin", F=(0.0, 1.0), CR=0.9)
NGEN = 300
SEED = 1
termination_multi = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-8,
    ftol=1e-8,
    period=20,
    n_max_gen=NGEN,
)

Myproblem = TwoObjExample()
res = minimize(
    Myproblem,
    gde3,
    termination_multi,
    seed=SEED,
    save_history=True,
    verbose=True
)
# popsize = 20
# algorithm = PSO(pop_size=20, max_iter=300)
#
# res = minimize(Myproblem,
#                algorithm,
#                seed=1,
#                verbose=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))  # mse and rmse
end = res.F[:]
min_index = np.argmin(end)
parama = res.X
print(parama)
OrderedDict_temp = OrderedDict()

for i in range(10):
    OrderedDict_temp[keys_list[i]] = inverse_min_max_scaling_single(parama[i], range_list[i][0], range_list[i][1])

with open(file_path, 'w') as file:
    file.write("*****************************************************************\n")
    file.write(".model  nmosmodelcard  nmos  level = 54 version = 4.0 \n")
    for param_name, param_value in OrderedDict_temp.items():
        file.write(f"+{param_name}={param_value}\n")

figure, ax = plt.subplots(figsize=(6, 5))
Vdd = 1.5
VGS = []
for i in range(6, 17, 2):
    VGS.append((i / 10))
for i in VGS:
    circuit = Circuit('NMOS Transistor')
    circuit.include(spice_library['nmosmodelcard'])
    Vgate = circuit.V('gate', 'gatenode', circuit.gnd, i @ u_V)
    Vdrain = circuit.V('drain', 'vdd', circuit.gnd, 0 @ u_V)
    # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
    circuit.MOSFET(1, 'vdd', 'gatenode', circuit.gnd, circuit.gnd, model='nmosmodelcard')
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vdrain=slice(0, Vdd, .1))
    ax.plot(analysis['vdd'], u_mA(-analysis.Vdrain), marker='o', linewidth=1,
            markersize=3)

ax.grid()
ax.set_xlabel('Vds [V]')
ax.set_ylabel('Id [mA]')
plt.tight_layout()
plt.savefig('Id_Vds.png')
plt.show()
plt.plot(np.log(convert))
np.save('Data2/DE.npy',np.log(convert))
plt.show()
