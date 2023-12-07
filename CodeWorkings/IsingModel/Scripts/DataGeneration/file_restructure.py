import numpy as np
import os
from datetime import datetime

def date_file(file_name):
    """
    Prepends the current date to a given file name.

    Parameters:
    - file_name (str): The name of the file.

    Returns:
    - str: A string containing the current date followed by the file name.
    """
    # Get the current date in 'YYYY-MM-DD' format
    date = datetime.today().strftime('%Y-%m-%d')
    # Concatenate the date and file name
    dated_name = str(str(date) + " " + file_name)
    return dated_name

def save_numpy_array(filename, array, folder_path):
    """
    Save a NumPy array to a specified location with a given name.

    Parameters:
    - filename (str): The name to be given to the saved file.
    - array (numpy.ndarray): The NumPy array to be saved.
    - folder_path (str): The path to the folder where the array will be saved.

    Returns:
    - str: The absolute file path of the saved NumPy array.
    """
    # Construct the absolute file path
    name = folder_path + filename + ".npy"
    # Open the file in binary write mode and save the array
    with open(name, mode="wb") as f:
        np.save(f, array)
    return name

def make_directory(folder_name, folder_path, topdir="../../"):
    """
    Create a folder in a specified location and return its path.

    Parameters:
    - folder_name (str): The name of the new folder.
    - folder_path (str): The path where the new folder will be created.
    - topdir (str, optional): The path to the uppermost directory needed to view the full project.

    Returns:
    - str: The absolute path of the created folder.
    """
    # Construct the absolute path of the new folder
    path = topdir + folder_path + folder_name
    # Create the folder if it doesn't exist
    os.makedirs(path, exist_ok=True)
    return path + "/"

def load_numpy_array(array_file_path):
    """
    Load a specified NumPy array and return the data contained in the array.

    Parameters:
    - array_file_path (str): The absolute path of the NumPy array file.

    Returns:
    - numpy.ndarray: The data contained in the loaded NumPy array.
    """
    # Load the NumPy array from the specified file path
    data = np.load(array_file_path, allow_pickle=True)
    return data

def delete_file(file_path):
    """
    Delete a file at the specified file path.

    Parameters:
    - file_path (str): The absolute path of the file to be deleted.
    """
    # Check if the file exists before attempting to delete
    if os.path.exists(file_path):
        os.remove(file_path)
    return
