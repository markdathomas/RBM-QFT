from CodeWorkings.IsingModel.Scripts.DataGeneration.data_generation_playground import run_and_plot



number_of_epochs = 4
alpha_size_list = [1*2**(-i) for i in range(number_of_epochs)]
N_repeats = 1
m_visible = 10
n_hidden = 4
batch_size = 100
save = True
steps_per_epoch = 3*10**1

print("Waiting for " + str(number_of_epochs) +  " epochs to run.")
run_and_plot(m_visible, n_hidden, alpha_size_list, batch_size, N_repeats, steps_per_epoch)