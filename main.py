from CodeWorkings.IsingModel.Scripts.DataGeneration.data_generation_playground import run_and_plot


#%% Edit these parameters
number_of_epochs = 4        #Number of different alpha values used. (Alpha = learning rate)
initial_alpha = 1           #The first alpha value used.
alpha_ratio = 2             #Inverse of common ratio of alpha values used 
N_repeats = 1               #Number of times each alpha value is run.
m_visible = 9               #Number of lattice sites.
n_hidden = 9                #Number of hidden nodes used
batch_size = 2            #Number of data points used in each training set.    
steps_per_epoch = 3*10**0   #Number of steps performed in each training epoch.


#%% Ignore this:
alpha_size_list = [initial_alpha*alpha_ratio**(-i) for i in range(number_of_epochs)]
print()
print("Waiting for " + str(number_of_epochs*N_repeats) +  " epochs to run.")
print("Number of Ising configurations possible: ", 2**m_visible)
print("Number of Ising configurations seen: ", number_of_epochs*N_repeats*batch_size*steps_per_epoch)
print()
run_and_plot(m_visible, n_hidden, alpha_size_list, batch_size, N_repeats, steps_per_epoch)