import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging
import numpy as np
logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from matplotlib import rcParams
config = {
        "font.family": 'Times New Roman',
        "axes.unicode_minus": False
    }
rcParams.update(config)
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.viridis(np.linspace(0, 1, 5))





libraries_path = "C:/PySpice/examples/libraries"
spice_library = SpiceLibrary(libraries_path)

Vdd = 1.5
VGS = []
for i in range(6, 17, 2):
    VGS.append((i / 10))
sim_re = []
for i in VGS:

    circuit = Circuit('NMOS Transistor')
    circuit.include(spice_library['bsim4'])
    Vgate = circuit.V('gate', 'gatenode', circuit.gnd, i @ u_V)
    Vdrain = circuit.V('drain', 'vdd', circuit.gnd, 0 @ u_V)
    # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
    circuit.MOSFET(1, 'vdd', 'gatenode', circuit.gnd, circuit.gnd, model='bsim4')
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vdrain=slice(0, Vdd, .1))
    # ax.plot(analysis['vdd'], u_mA(-analysis.Vdrain), marker='o', linewidth=1,
    #         markersize=3)
    ax.scatter(np.array(analysis['vdd']).tolist(), np.array(u_mA(-analysis.Vdrain)).tolist(), color=colors[0], marker='o', edgecolors='black', s=100, label='Measured')
    Sim_Vds = {}
    ax.plot(analysis['vdd'], u_mA(-analysis.Vdrain), marker='*', linewidth=3, markersize=3, color=colors[2], label='Modeled')
    Sim_Vds['Vds'] = np.array(analysis['vdd']).tolist()
    Sim_Vds['Id'] = np.array(u_mA(-analysis.Vdrain)).tolist()
    Sim_Vds['Vgs'] = i
    sim_re.append(Sim_Vds)

ax.set_xlabel('Vds [V]')
ax.set_ylabel('Id [mA]')
ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.tick_params(axis='both', which='both', direction='out', width=0.5)
plt.show()

Id=[]
for i in range(len(sim_re)):
    Id.append(sim_re[i]['Id'])
Id=np.array(Id)
# save npy
print(Id.shape)
np.save('../Data/Id.npy', Id)





