

import numpy as np

from collections import OrderedDict

ordered_dict = OrderedDict()
ordered_dict['tnom'] = 27
ordered_dict['epsrox'] = 3.9
ordered_dict['eta0'] = 0.0058
ordered_dict['nfactor'] = 1.9
ordered_dict['wint'] = 5e-09
ordered_dict['cgso'] = 1.5e-10
ordered_dict['cgdo'] = 1.5e-10
ordered_dict['xl'] = -3e-08
ordered_dict['toxe'] = 1.85e-09
ordered_dict['toxp'] = 1.2e-09
ordered_dict['toxm'] = 1.85e-09
ordered_dict['toxref'] = 1.85e-09
ordered_dict['dtox'] = 6.5e-10
ordered_dict['lint'] = 5.25e-09
ordered_dict['vth0'] = 0.429
ordered_dict['k1'] = 0.497
ordered_dict['u0'] = 0.04861
ordered_dict['vsat'] = 124340
ordered_dict['rdsw'] = 165
ordered_dict['ndep'] = 2.6e+18
ordered_dict['xj'] = 1.96e-08
keys_list = list(ordered_dict.keys())
range_list = [(value * 0.5, value * 1.5) if isinstance(value, (int, float)) else (None, None) for value in
              ordered_dict.values()]

def inverse_scaling_single(scaled_value, original_min, original_max):
    return scaled_value * (original_max - original_min)
