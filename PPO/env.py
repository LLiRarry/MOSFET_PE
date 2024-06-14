import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# this is costomer env for parameter explore
from utils import range_list, keys_list, OrderedDict, ordered_dict,inverse_scaling_single
import PySpice.Logging.Logging as Logging

logger = Logging.setup_logging()
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from sklearn.decomposition import PCA
libraries_path = "C:/PySpice/examples/libraries"
spice_library = SpiceLibrary(libraries_path)

class envs():
    def __init__(self):
        self.observation_space_shape = 21  # 1+20
        self.action_space_shape = 21  #21 +step_size
        self.seed = 4008
        self.Parameter = ordered_dict
        self.keys_list = keys_list
        self.Ground = np.load('../Data/Id.npy')


    def reset(self):  # state

        param_state = []
        for key, value in self.Parameter.items():
            self.Parameter[key] = np.random.uniform(range_list[keys_list.index(key)][0],
                                                    range_list[keys_list.index(key)][1])
            param_state.append(value)
        simulation_rmse,reduced_data = self.simulatetion(self.Parameter)

        state=np.concatenate((simulation_rmse,reduced_data),axis=0)
        return state

    def simulatetion(self, Params):
        def chage_lib_file():
            with open('C:\\PySpice\\examples\\libraries\\mosfet\\nmosmodelcard.lib', 'w') as file:
                file.write("*****************************************************************\n")
                file.write(".model  nmosmodelcard  nmos  level = 54 version = 4.0 \n")
                for param_name, param_value in Params.items():
                    file.write(f"+{param_name}={param_value}\n")

        chage_lib_file()
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
            Sim_Vds = {}
            Sim_Vds['Vds'] = np.array(analysis['vdd']).tolist()
            Sim_Vds['Id'] = np.array(u_mA(-analysis.Vdrain)).tolist()
            Sim_Vds['Vgs'] = i
            sim_re.append(Sim_Vds)
        Id = []
        for i in range(len(sim_re)):
            Id.append(sim_re[i]['Id'])
        Id = np.array(Id).flatten()
        GroundTruth = self.Ground.flatten()
        rmse = np.array([np.sqrt(np.mean((Id - GroundTruth) ** 2))])
        return rmse

    def step(self, action):
        # action=action.detach().numpy()
        for i in range(len(action)):
            self.Parameter[self.keys_list[i]] += inverse_scaling_single(action[i], range_list[i][0], range_list[i][1])
            self.Parameter[self.keys_list[i]] = np.clip(self.Parameter[self.keys_list[i]], range_list[i][0], range_list[i][1])
        # print(action,"********action********")
        simulation_rmse,reduce_data = self.simulatetion(self.Parameter)
        # print(simulation_rmse,"********simulation_rmse********")
        state = np.concatenate((simulation_rmse, reduce_data), axis=0)
        reward = simulation_rmse[0] * (-1)
        # reward=np.exp(-1*reward)-1
        thold = 0.0001
        done = False
        if simulation_rmse[0] < thold:
            done = True
        return state, reward, done
