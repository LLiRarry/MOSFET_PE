import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging
import numpy as np
logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
libraries_path = "C:/PySpice/examples/libraries"
spice_library = SpiceLibrary(libraries_path)
figure, ax = plt.subplots(figsize=(6, 5))
Vdd = 1.1
VGS = []
for i in range(0, 20, 1):
    VGS.append((i / 20))
sim_re = []
for i in VGS:
    circuit = Circuit('NMOS Transistor')
    circuit.include(spice_library['nmosmodelcard'])
    Vgate = circuit.V('gate', 'gatenode', circuit.gnd, i @ u_V)
    Vdrain = circuit.V('drain', 'vdd', circuit.gnd, 0 @ u_V)
    # M <name> <drain node> <gate node> <source node> <bulk/substrate node>
    circuit.MOSFET(1, 'vdd', 'gatenode', circuit.gnd, circuit.gnd, model='nmosmodelcard')
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.dc(Vdrain=slice(0, Vdd, .05))
    ax.plot(analysis['vdd'], u_mA(-analysis.Vdrain), marker='o', linewidth=1,
            markersize=3)
    Sim_Vds = {}
    Sim_Vds['Vds'] = np.array(analysis['vdd']).tolist()
    Sim_Vds['Id'] = np.array(u_mA(-analysis.Vdrain)).tolist()
    Sim_Vds['Vgs'] = i
    sim_re.append(Sim_Vds)
ax.grid()
ax.set_xlabel('Vds [V]')
ax.set_ylabel('Id [mA]')
plt.tight_layout()
plt.savefig('Id_Vds2.png')
plt.show()
Id=[]
for i in range(len(sim_re)):
    Id.append(sim_re[i]['Id'])
Id=np.array(Id)
# save npy
print(Id.shape)
np.save('../Data/Id.npy', Id)





