from CodeWorkings.IsingModel.Scripts.DataGeneration.data_generation_playground import run_and_plot


#%% Edit these parameters
number_of_epochs = 3            #Number of different alpha values used. (Alpha = learning rate)
initial_alpha = 0.01           #The first alpha value used.
alpha_ratio = 10             #Inverse of common ratio of alpha values used 
N_repeats = 1               #Number of times each alpha value is run.
m_visible = 6               #Number of lattice sites. 
n_hidden = 6                #Number of hidden nodes used
batch_size = 200              #Number of data points used in each training set.    
steps_per_epoch = 1*10**3   #Number of steps performed in each training epoch.
double_first_epoch = True  #Perform the first run twice? (as in Cossu et al). 

run_and_plot(m_visible, n_hidden, initial_alpha, alpha_ratio, batch_size, N_repeats, steps_per_epoch, double_first_epoch, number_of_epochs)