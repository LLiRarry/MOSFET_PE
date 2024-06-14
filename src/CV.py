import math

import matplotlib.pyplot as plt
import numpy as np
import PySpice.Logging.Logging as Logging
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()

libraries_path = "C:/PySpice/examples/libraries"
spice_library = SpiceLibrary(libraries_path)

gate_voltage = np.linspace(-5, 5, num=50)

capacitance_values = []
import numpy as np


def calculate_rms(current_waveform):
    rms = np.sqrt(np.mean(np.square(current_waveform)))
    return rms


# 执行不同门源电压下的仿真并测量电容
for vg in gate_voltage:
    print(vg)
    # 首先仿真Cgs-Vgs
    circuit = Circuit('MOSFET Capacitance-Voltage')
    circuit.include(spice_library['bsim4'])
    M1 = circuit.MOSFET(1, 'drain', 'gate', 'source', 'bulk', model='bsim4')
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Vgate= circuit.V('gate_in', 'gate', circuit.gnd, vg @ u_V)
    Vdrain = circuit.V('drain', 'drain', circuit.gnd, 0 @ u_V)
    ac_line = circuit.AcLine('ac_scale', 'gate', 'source', rms_voltage=1.0 @ u_V,
                             frequency=1 @ u_MHz)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    analysis = simulator.transient(step_time=0.001 @ u_ns, end_time=ac_line.period * 2)
    print('******************************************')
    print(u_A(-analysis.gate))
    I = abs(u_A(-analysis.gate)) / (2 * np.pi * ac_line.frequency)
    rms_I = calculate_rms(I)
    capacitance_values.append(rms_I)
    #
    # capacitance_values.append(abs(analysis[M1.gate, M1.drain]) * u_F)

plt.figure(figsize=(8, 6))
plt.plot(gate_voltage, capacitance_values)
plt.title('MOSFET Capacitance-Voltage')
plt.xlabel('Gate-Source Voltage [V]')
plt.ylabel('Capacitance [F]')
plt.grid()
plt.tight_layout()
plt.show()
