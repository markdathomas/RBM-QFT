import numpy as np
import os
from datetime import datetime

def date_file(file_name):
    """Puts the data on the front of the folder and title after"""
    date = datetime.today().strftime('%Y-%m-%d')
    dated_name = str(str(date)+" " + file_name)
    return dated_name

def save_numpy_array(filename, array, folder_path):
    """Takes in a numpy array and saves it in the specified location
    with the given name
    
    Example:
    #Save data to the new folder
    saving_array = np.array([1,2,3]) #Array to be saved

    array = "lunchtime3" #Giving the array a name
    array_name = date_file(array) #Dating the array name for the file

    #Save the array
    saved_array_file_path = save_numpy_array(array_name, saving_array, folder_path)
    """
    
    name = folder_path+ filename +".npy"
    with open(name, mode="wb") as f:
        np.save(f,array)
    return name

def make_directory(folder_name, folder_path, topdir = "../../"):
    """Create a folder in a specified location and returns the path of that folder

    topdir is the path to the uppermost directory needed to view the full project.
    
    To make a directory and find the folder path, use the following example:
    new_folder_absolute_path = "Data/"
    new_folder_name = "new folder"
    make_directory(new_folder_name,new_folder_absolute_path, topdir)
    folder_path = topdir+new_folder_absolute_path+new_folder_name+"/"
    """
    path = topdir+folder_path+folder_name

    os.makedirs(path, exist_ok=True)
    return  path+"/"

def load_numpy_array(array_file_path):
    """Load a specified numpy array, returns the data contained in the array.
    
    Example:    
    b = load_numpy_array(saved_array_file_path), where saved_array_file_path 
    defined in save_numpy_array function example.
    """
    data = np.load(array_file_path, allow_pickle = True)
    return data

def delete_file(file_path):
    """
    Deletes the file in the filepath mentioned
    
    Example:
    delete_file(saved_array_file_path)
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    return
