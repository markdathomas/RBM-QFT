# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:00:42 2021

@author: kramm
"""
import numpy as np
import sys

sys.path.insert(0, "../Data_generation")
try:
    from file_restructure import load_numpy_array
except ImportError:
    print('No Import')

#Load data from the new folder



def load_data(folder_date, data_date, data_params, data_name):
    if data_date == None:
        file_to_load = "Data/" + folder_date +" " + data_params + "/" + data_name + ".npy"
    else:
        file_to_load = "Data/" + folder_date +" " + data_params + "/" + data_date + " " + data_name + ".npy"
    loaded_array = load_numpy_array(file_to_load)
    return loaded_array



#Example usage
#data_date = "2021-11-24"
#data_name = "m6_n6_b20"
#../../Data/2021-11-24 m 6 n 6b 200 [200, 100, 100][0.01, 0.001, 0.0001]/2021-11-24 m 6 n 6b 200 [200, 100, 100][0.01, 0.001, 0.0001].npy

#data_name = "m 6 n 6b 200 [200, 100, 100][0.01, 0.001, 0.0001]"
#data_date = "2021-11-24"

#array = load_data(data_date, data_name)
#print(len(array))

"""
#Delete the file that held the new data
delete_file(saved_array_file_path)
"""
