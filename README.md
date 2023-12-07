# Learning the Ising Model with RBMs

## Description
The "Learning the Ising Model with RBMs" project simulates a 1D Ising model lattice and trains a restricted Boltzmann machine (RBM) on it using the contrastive divergence algorithm.

## Getting started (on MacOS)
Start by setting up an environment in command/terminal. For example, to generate an environment called `rbmenv` use
    python -m venv rbmenv

Once this is done, activate the environment using
    source rbmenv/bin/activate

Next install the requirements using 
    pip install -r requirements.txt

Once this is done you're all ready to go. Simply tweak the main.py file as instructed below, then to run it simply execute
    python main.py

## Parameters
This section provides users with information on how to customize the parameters and what each parameter represents. Adjust it as needed based on your project's specifics.

To customize the behavior of the simulation and RBM training, you can edit the following parameters in the `main.py` file:

- `number_of_epochs`: Number of different alpha values used. (Alpha = learning rate)
- `initial_alpha`: The first alpha value used.
- `alpha_ratio`: Inverse of the common ratio of alpha values used.
- `N_repeats`: Number of times each alpha value is run.
- `m_visible`: Number of lattice sites.
- `n_hidden`: Number of hidden nodes used.
- `batch_size`: Number of data points used in each training set.
- `steps_per_epoch`: Number of steps performed in each training epoch.


# For example:
number_of_epochs = 4
initial_alpha = 1
alpha_ratio = 2
N_repeats = 1
m_visible = 6
n_hidden = 6
batch_size = 200
steps_per_epoch = 3*10**3



The generated data and plots will be saved in folders with the date the file was generated and the run parameters used.

Feel free to experiment with different parameter values to observe their impact on the simulation and RBM training results.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.