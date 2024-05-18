INIT_PROBABILITY = 0.08
INHERTIANCE_PROBABILITY  = 0.300 # probability that neighboring cells will inherit by perturbation.
NUM_LAYERS = 3 # rest hidden and one alpha
parameter_perturbation_probability = 0.20
budget_per_cell = 3
activation_on_board = ['sigmoid'] # sigmoid or tanh

# Rest of the Code
def print(*args, **kwargs):
    # Uncomment the next line to enable printing
    # built_in_print(*args, **kwargs)
    pass
import torch
from PIL import Image
import shutil
import os
import numpy as np


if not os.path.exists('NCA'):
    os.makedirs('NCA')

if not os.path.exists('PD'):
    os.makedirs('PD')

if not os.path.exists('GD'):
    os.makedirs('GD')

import time
output_stamp = int(time.time())
if not os.path.exists('Outputs_'+str(output_stamp)):
    os.makedirs('Outputs_'+str(output_stamp))

import sys
sys.setrecursionlimit(10**6)

precision = 1
torch.set_printoptions(precision=precision)
WIDTH, HEIGHT = 50,50
grid_size = (WIDTH, HEIGHT)
print("Width and Height used are {} and {}".format(WIDTH, HEIGHT))
min_pixels = max(0, int(WIDTH * HEIGHT * INIT_PROBABILITY))
ALPHA = 0.5 # To make other cells active (we dont go with other values below 0.6 to avoid dead cells and premature livelihood)
print("Numbers of layers used are {}".format(NUM_LAYERS))
print("1 for alpha layer and rest {} for hidden".format(NUM_LAYERS-1))
NUM_STEPS = 300
num_steps = NUM_STEPS
at_which_step_random_death = 9999999999 # Set this to infinity or high value if you never want to enter catastrophic deletion (random death happens at this generation)
probability_death = 0.004 # 40 pixels die every generation
print("Numbers of Time Steps are {}".format(NUM_STEPS))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
activation = 'sigmoid' # ['sigmoid','tanh','noact']
frequency_dicts = []
FPS = 10 # Speed of display for animation of NCA and plots
marker_size = 1 # for plots
everystep_weights = [] # Stores weigths of the NNs from every time step.
KMEANS_K = 5
enable_annotations_on_nca = True


fixed_value = 0
budget_counter_grid = np.zeros((WIDTH, HEIGHT)) + fixed_value



import torch
import time
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import random
import copy
from collections import Counter
from matplotlib.colors import Normalize

ca_grids_for_later_analysis = []

# def custom_activation(x):
#     result = 0.05 * (x + 2)
#     result = max(0, min(0.1, result))
#     return result
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    updated_value = np.tanh(x)
    updated_value_tensor = torch.tensor(updated_value, dtype=torch.float32)
    return updated_value_tensor
if(activation == 'sigmoid'):
  class SimpleNN(nn.Module):
      def __init__(self):
          super(SimpleNN, self).__init__()
          self.fc1 = nn.Linear(9 * NUM_LAYERS, NUM_LAYERS)
          self.sigmoid = nn.Sigmoid()
          self.fc2 = nn.Linear(NUM_LAYERS, NUM_LAYERS)
          # Initialize weights and biases to zero
          nn.init.zeros_(self.fc1.weight)
          nn.init.zeros_(self.fc2.weight)
          nn.init.zeros_(self.fc1.bias)
          nn.init.zeros_(self.fc2.bias)
      def forward(self, x):
          x = self.fc1(x)
          x = self.sigmoid(x)
          x = self.fc2(x)
          return x

elif(activation == 'tanh'):
  class SimpleNN(nn.Module):
      def __init__(self):
          super(SimpleNN, self).__init__()
          self.fc1 = nn.Linear(9 * NUM_LAYERS, NUM_LAYERS)
          self.tanh = nn.Tanh()
          self.fc2 = nn.Linear(NUM_LAYERS, NUM_LAYERS)
          # Initialize weights and biases to zero
          nn.init.zeros_(self.fc1.weight)
          nn.init.zeros_(self.fc2.weight)
          nn.init.zeros_(self.fc1.bias)
          nn.init.zeros_(self.fc2.bias)
      def forward(self, x):
          x = self.fc1(x)
          x = self.tanh(x)
          x = self.fc2(x)
          return x

elif(activation == 'noact'):
  class SimpleNN(nn.Module):
      def __init__(self):
          super(SimpleNN, self).__init__()
          self.fc1 = nn.Linear(9 * NUM_LAYERS, NUM_LAYERS)
          self.fc2 = nn.Linear(NUM_LAYERS, NUM_LAYERS)
          # Initialize weights and biases to zero
          nn.init.zeros_(self.fc1.weight)
          nn.init.zeros_(self.fc2.weight)
          nn.init.zeros_(self.fc1.bias)
          nn.init.zeros_(self.fc2.bias)
      def forward(self, x):
          x = self.fc1(x)
          x = self.fc2(x)
          return x


elif(activation == 'relu'):
  class SimpleNN(nn.Module):
      def __init__(self):
          super(SimpleNN, self).__init__()
          self.fc1 = nn.Linear(9 * NUM_LAYERS, NUM_LAYERS)
          self.relu = nn.ReLU()
          self.fc2 = nn.Linear(NUM_LAYERS, NUM_LAYERS)
          # Initialize weights and biases to zero
          nn.init.zeros_(self.fc1.weight)
          nn.init.zeros_(self.fc2.weight)
          nn.init.zeros_(self.fc1.bias)
          nn.init.zeros_(self.fc2.bias)
      def forward(self, x):
          x = self.fc1(x)
          x = self.relu(x)
          x = self.fc2(x)
          return x
else:
  class SimpleNN(nn.Module):
      def __init__(self):
          super(SimpleNN, self).__init__()
          self.fc1 = nn.Linear(9 * NUM_LAYERS, NUM_LAYERS)
          self.leaky_relu = nn.LeakyReLU()
          self.fc2 = nn.Linear(NUM_LAYERS, NUM_LAYERS)
          # Initialize weights and biases to zero
          nn.init.zeros_(self.fc1.weight)
          nn.init.zeros_(self.fc2.weight)
          nn.init.zeros_(self.fc1.bias)
          nn.init.zeros_(self.fc2.bias)
      def forward(self, x):
          x = self.fc1(x)
          x = self.leaky_relu(x)
          x = self.fc2(x)
          return x


ca_nn = SimpleNN().to(DEVICE)

# Calculate the number of parameters stepwise

input_size_fc1 = 9 * NUM_LAYERS
output_size_fc1 = NUM_LAYERS
weight_params_fc1 = input_size_fc1 * output_size_fc1
bias_params_fc1 = output_size_fc1

input_size_fc2 = NUM_LAYERS
output_size_fc2 = NUM_LAYERS
weight_params_fc2 = input_size_fc2 * output_size_fc2
bias_params_fc2 = output_size_fc2

weight_params = weight_params_fc1 + weight_params_fc2
bias_params = bias_params_fc1 + bias_params_fc2
total_params = weight_params + bias_params

print("------------------------------------------------------------------")
print("Model Summary")
# Print the equations and the total number of parameters
print(f"Input Size: {input_size_fc1}")
print(f"FC Layer 1 Weight Parameters: {input_size_fc1} * {output_size_fc1} = {weight_params_fc1}")
print(f"FC Layer 1 Bias Parameters: {output_size_fc1}")
print(f"FC Layer 2 Weight Parameters: {input_size_fc2} * {output_size_fc2} = {weight_params_fc2}")
print(f"FC Layer 2 Bias Parameters: {output_size_fc2}")
print(f"Total Number of Parameters: {total_params}")
for name, layer in ca_nn.named_children():
    if isinstance(layer, nn.Linear):
        input_size = layer.in_features
        output_size = layer.out_features
        if name == "fc1":
            print(f"Layer 1: {name}, Input Size: {input_size}, Output Size: {output_size_fc1}")
        elif name == "fc2":
            print(f"Layer 2: {name}, Input Size: {input_size_fc2}, Output Size: {output_size_fc2}")

print("------------------------------------------------------------------")


ca_grid = torch.zeros((NUM_LAYERS, WIDTH, HEIGHT), device=DEVICE, dtype=torch.float32)
print("cagrid before")
print(ca_grid)
random_tensor = torch.ones((WIDTH, HEIGHT))
sorted_values, indices = random_tensor.view(-1).sort()
mask = torch.zeros_like(random_tensor)
shuffled_indices = indices.tolist()  # Convert indices to a list
random.shuffle(shuffled_indices)     # Shuffle the list
random_positions = shuffled_indices[:min_pixels]
mask.view(-1)[random_positions] = 1.0
for layer in range(0,NUM_LAYERS):
  if (layer==0):
    ca_grid[layer] = mask * (ALPHA + (1 - ALPHA) * random_tensor)  # Initialize the alpha channel with values greater than ALPHA
  else:
    ca_grid[layer] = mask * (ALPHA + (1 - ALPHA) * random_tensor)  # Initialize the other channel with values greater than ALPHA
# ca_grid[0] = mask * (random_tensor) # Initialising only alpha channel
# ca_grid[1] = mask * (random_tensor * random.random()) # Initialising a little amount of channel 1 with the idea that if a pixel is having some intensity in alpha, it should have some values in other channels.
print("cagrid after")
print(ca_grid)
print("------------------------------------------------------------------")

print("Weight Parameters:")

print("Individual Weight Parameters:")
for name, param in ca_nn.named_parameters():
    if 'weight' in name:
        print(f"Layer: {name}, Shape: {param.shape}")
        print(f"Weights: {param}")

print("------------------------------------------------------------------")
print("Bias Parameters for FC1:")
print(ca_nn.fc1.bias)
print("Bias Parameters for FC2:")
print(ca_nn.fc2.bias)

print("------------------------------------------------------------------")

print("Entering update loop >>>>>")

def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
def initialize_weights_to_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)

# Create a list of neural networks, one for each pixel
ca_nn_list = [SimpleNN().to(DEVICE) for _ in range(WIDTH * HEIGHT)]

# Initialize the weights for neural networks associated with live pixels
# budget_per_cell = 8
# fixed_value = 0
# budget_counter_grid = np.zeros((WIDTH, HEIGHT)) + fixed_value


for i in range(WIDTH):
    for j in range(HEIGHT):
        if ca_grid[0, i, j] > ALPHA:
            ca_nn_list[i * WIDTH + j].apply(initialize_weights)

def update_ca(ca_grid, ca_nn_list,frame_number):
    print("")
    print("")

    print("Inside updateCA function")
    new_ca_grid = ca_grid.clone()
    print("New CA grid initialised temporarily:")
    print(new_ca_grid)
    just_inherited_indices = []
    def process_neighborhood(i, j, idx, ca_nn_list):
        print("")
        print("")
        print("Inside process_neighborhood function")
        neighborhood = torch.zeros(9 * NUM_LAYERS, device=DEVICE)
        print("Empty tensor for neighborhood initialised temporarily:")
        print(neighborhood)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                ni, nj = (i + dx) % WIDTH, (j + dy) % HEIGHT
                for l in range(NUM_LAYERS):
                    neighborhood[(dx + 1) * 3 + (dy + 1) + l * 9] = ca_grid[l, ni, nj]
        neighborhood = torch.unsqueeze(neighborhood, 0)
        print("Neighborhood is :")
        print(neighborhood)
        output = None  # Initialize output to None
        print("------------------------------------------------------------------")

        print("Weight Parameters at this step (no perturbation has happened at this step):")

        print("Individual Weight Parameters:")
        for name, param in ca_nn_list[idx].named_parameters():
            if 'weight' in name:
                print(f"Layer: {name}, Shape: {param.shape}")
                print(f"Weights: {param}")


        print("------------------------------------------------------------------")
        print("Bias Parameters at this step for FC1:")
        print(ca_nn_list[idx].fc1.bias)
        print("Bias Parameters at this step for FC2:")
        print(ca_nn_list[idx].fc2.bias)

        print("------------------------------------------------------------------")
        if((neighborhood > ALPHA).any()):
          output = ca_nn_list[idx](neighborhood)
          print("🙋Inside if condition for (neighborhood > ALPHA).any() ")
          if(random.random() < INHERTIANCE_PROBABILITY):
            print("🙋🙋Reaching towards inheritance and performing pertubation too - inside if condition - random.random() < INHERTIANCE_PROBABILITY.")
            high_alpha_pixels = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if ca_grid[0, (i + dx) % WIDTH, (j + dy) % HEIGHT] > ALPHA] # Find neighboring pixels with alpha values greater than ALPHA
            if high_alpha_pixels:
                # check again for livelihood of the pixels (if this if condition passed, that means there is at least one element that is greater than alpha.)
                current_alpha_value = ca_grid[0, i, j]
                print("current_alpha_value: ",current_alpha_value.item())
                print("current_alpha_value: ",current_alpha_value.item())
                print("current_alpha_value: ",current_alpha_value.item())
                print("current_alpha_value: ",current_alpha_value.item())
                high_alpha_values = [ca_grid[0, (i + dx) % WIDTH, (j + dy) % HEIGHT] for dx, dy in high_alpha_pixels]
                print(">>>> Hello! I am pixel at position {}, {}. My alive neighbors are at the positions {} whose values in Alpha channel are {}.".format(i, j, high_alpha_pixels, high_alpha_values))
                selected_pixel = random.choice(high_alpha_pixels)  # Select any random live pixel in the neighborhood
                print("I am selecting pixel {}".format(selected_pixel))
                ni, nj = (i + selected_pixel[0]) % WIDTH, (j + selected_pixel[1]) % HEIGHT

                # Check if the selected neural network has any non-zero weights
                selected_nn_idx = ni * WIDTH + nj
                selected_nn = ca_nn_list[selected_nn_idx]
                has_nonzero_weights_selected = any(torch.any(param != 0) for param in selected_nn.parameters())

                current_nn_idx = idx
                current_nn = ca_nn_list[current_nn_idx]
                has_nonzero_weights_current = any(torch.any(param != 0) for param in current_nn.parameters())

                if not has_nonzero_weights_current:
                    just_inherited_indices.append(idx)
                    # Copy the neural network from the selected neighboring pixel
                    ca_nn_list[idx] = copy.deepcopy(selected_nn)
                    print("AM I EVEN ENTERING HERE")
                    print("AM I EVEN ENTERING HERE")
                    print("AM I EVEN ENTERING HERE")
                    print("AM I EVEN ENTERING HERE")
                    # Perturb neural network weights
                    for name, param in ca_nn_list[idx].named_parameters():
                        if 'weight' in name:
                            mask = torch.rand_like(param.data) < parameter_perturbation_probability
                            with torch.no_grad():
                                param.data += mask * (torch.rand_like(param.data) - 0.5) * 2 # Achieves a value between -1 to 1
                else:
                    ca_nn_list[idx] = copy.deepcopy(selected_nn)
        else:
          ca_nn_list[idx].apply(initialize_weights_to_zero)
        can_nn_updated_single = copy.deepcopy(ca_nn_list[idx])
        print("Ouptut from NN is :")
        print(output)
        print("------------------------------------------------------------------")

        print("Weight Parameters at this step (if perturbation, then it has already finished at this step):")

        print("Individual Weight Parameters:")
        for name, param in ca_nn_list[idx].named_parameters():
            if 'weight' in name:
                print(f"Layer: {name}, Shape: {param.shape}")
                print(f"Weights: {param}")

        print("------------------------------------------------------------------")
        print("Bias Parameters at this step for FC1:")
        print(ca_nn_list[idx].fc1.bias)
        print("Bias Parameters at this step for FC2:")
        print(ca_nn_list[idx].fc2.bias)

        print("------------------------------------------------------------------")

        if output is not None:
            # print("returning output from this function: this 1 should be same as this 2", output.squeeze().tolist())
            return output.squeeze().tolist(), can_nn_updated_single
        else:
            # Return a default value (all zeros) if output is None
            return [0.0] * NUM_LAYERS, can_nn_updated_single
    print("------------------------------------Scanning each cell------------------------------------")
    idx = 0  # Index for the neural networks
    ca_nn_list_temp = []
    
    for i in range(WIDTH):
        for j in range(HEIGHT):
            # print("------------------------------------------------------------------")
            # print(">>Right now I am using pixel at WIDTH {}, HEIGHT {}".format(i,j))
            updated_values, can_nn_updated_single = process_neighborhood(i, j, idx, ca_nn_list)
            # print("returning output from process_neighborhood function which is same as the ouptut of NN in high precision: ", updated_values)
            for layer in range(NUM_LAYERS):
                new_ca_grid[layer, i, j] = updated_values[layer]
            ca_nn_list_temp.append(can_nn_updated_single)
            idx += 1

    print("⏩⏩⏩Final updated grid :")
    print(new_ca_grid)
    print("⏩⏩⏩Final updated grid (and the values are set to 0 or dead for the pixels are below ALPHA.):")
    new_ca_grid_temp = new_ca_grid.clone()
    print("now replace all pixels with their corresponding sigmoid values.")
    # now replace all pixels with their corresponding sigmoid values.
    for x in range(WIDTH):
      for y in range(HEIGHT):
        for layer in range(NUM_LAYERS):
          if(activation_on_board == 'sigmoid'):
            updated_value = sigmoid(new_ca_grid_temp[layer, x, y].cpu().numpy())
          else:
            updated_value = tanh(new_ca_grid_temp[layer, x, y].cpu().numpy())
          new_ca_grid_temp[layer, x, y] = updated_value # setting a value between 0 and 1 for the alive pixels.
    print(new_ca_grid_temp)
    print("⏩⏩⏩now we will check for the ALPHA values as threshold")
    # now we will check for the ALPHA values as threshold
    for x in range(WIDTH):
      for y in range(HEIGHT):
          # Check if any value in channel 0 is less than ALPHA at the current position
          if (new_ca_grid_temp[0, x, y] <= ALPHA):
            # If any value is less than ALPHA, set values in all layers at the current position to 0
            for layer in range(NUM_LAYERS):
                new_ca_grid_temp[layer, x, y] = 0.0
    
    
    
    # Budget Counter Grid
    # Budget Counter Grid
    # Budget Counter Grid
    print("------------------------------------BUDGET COUNTER CHECKS------------------------------------")
    print("Budget Counter Grid - Before")
    print(budget_counter_grid)

    temp_budget_counter_grid_iterator = 0
    index_temp_budget_counter_grid_iterator = []
    for i in range(WIDTH):
        for j in range(HEIGHT):
            if round(new_ca_grid_temp[0, i, j].item(),precision) > ALPHA:
                print("Incrementing the counter with 1 for budget increment")
                budget_counter_grid[i,j] = budget_counter_grid[i,j] + 1 # Budget consumed per generation
            if round(new_ca_grid_temp[0, i, j].item(),precision) <= ALPHA:
                print("For dead cells making counter 0 to not carry the leaks (agents might have died in the update rule as well)")
                budget_counter_grid[i,j] = 0 # Counter to 0 all rest cases
                # index_temp_budget_counter_grid_iterator.append(temp_budget_counter_grid_iterator)
                # for layer in range(NUM_LAYERS):
                #     new_ca_grid_temp[layer, x, y] = 0.0
            temp_budget_counter_grid_iterator = temp_budget_counter_grid_iterator + 1

    if(len(index_temp_budget_counter_grid_iterator)>0):
        for ii in range(len(ca_nn_list_temp)):
            if ii in index_temp_budget_counter_grid_iterator:
                ca_nn_list_temp[ii].apply(initialize_weights_to_zero)
    
    print("Budget Counter Grid - After increment and dead cell deletion")
    print(budget_counter_grid)

    
    # Death Routine from previosu code but with budget counter grid condition
    counter_death_budget = 0
    index_death_collected_budget = []
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if (budget_counter_grid[x,y]>budget_per_cell): # check for budget limit
                # If any value is less than ALPHA, set values in all layers at the current position to 0
                print("Threshold reached Threshold reached Threshold reached!!!!")
                budget_counter_grid[x,y] = 0 # <<<<<<<RESET COUNTER>>>>>>>>>>>>
                index_death_collected_budget.append(counter_death_budget)
                for layer in range(NUM_LAYERS):
                    print("Resetting CHANNELS threshold reached")
                    new_ca_grid_temp[layer, x, y] = 0.0
            counter_death_budget = counter_death_budget + 1
                    
    print("Budget Counter Grid - After resetting in case it reaches threshold")
    print(budget_counter_grid)


    if(len(index_death_collected_budget)>0):
        for ii in range(len(ca_nn_list_temp)):
            if ii in index_death_collected_budget:
                ca_nn_list_temp[ii].apply(initialize_weights_to_zero)
    print("------------------------------------BUDGET COUNTER CHECKS------------------------------------")
    print(new_ca_grid_temp)

    print("------------------------------------MAKE FINAL CHECK TO REPLACE ALPHA WITH 0 and ANNs null------------------------------------")
    # for x in range(WIDTH):
    #     for y in range(HEIGHT):
    #         if (budget_counter_grid[x,y] == 0):
    #             for layer in range(NUM_LAYERS):
    #                 print("Resetting CHANNELS threshold reached")
    #                 new_ca_grid_temp[layer, x, y] = 0.0
    

    # Make ANNs also die for the same places (again!!!)
    final_annns_counter = 0
    idx_final_annns_counter = []
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if (budget_counter_grid[x,y] == 0):
                idx_final_annns_counter.append(final_annns_counter) 
                for layer in range(NUM_LAYERS):
                    print("Resetting CHANNELS threshold reached")
                    new_ca_grid_temp[layer, x, y] = 0.0
            final_annns_counter = final_annns_counter + 1

    if(len(idx_final_annns_counter)>0):
        for ii in range(len(ca_nn_list_temp)):
            if ii in idx_final_annns_counter:
                # Check to not make the ANNs that are just inherited and hence dont make them 0 accidently
                if ii not in just_inherited_indices: # This ensures that only make those ANNs to 0 that are not in the just_inherited_indices so that only dead ANNs gets removed and not the inherited ones (because just_inherited ANNs also has 0 alpha)
                    ca_nn_list_temp[ii].apply(initialize_weights_to_zero)
    return new_ca_grid_temp, ca_nn_list_temp

if not os.path.exists('sim_frames_png'):
    os.makedirs('sim_frames_png')
if not os.path.exists('sim_frames_pdf'):
    os.makedirs('sim_frames_pdf')



metadata = dict(title='Neural CA Simulation', artist='AI', comment='Neural CA Simulation')
writer = FFMpegWriter(fps=FPS, metadata=metadata)

all_colormaps = plt.colormaps()
# colormaps = all_colormaps
colormaps = ['magma']
fig, axes = plt.subplots(1, NUM_LAYERS, figsize=(5 * NUM_LAYERS, 5))
# plt.close(fig)
import time
stamp = int(time.time())
with writer.saving(fig, "NCA_video_{}.mp4".format(stamp), dpi=600):
    for frame in range(NUM_STEPS+1):
        fig, axes = plt.subplots(1, NUM_LAYERS, figsize=(5 * NUM_LAYERS, 5))
        if(NUM_LAYERS==2):
            cax = fig.add_axes([0.08, 0.94, 0.08, 0.02])  # Adjust the position and size as needed
        elif(NUM_LAYERS==3):
            cax = fig.add_axes([0.05, 0.955, 0.08, 0.02])  # Adjust the position and size as needed
        else:
            cax = fig.add_axes([0.02, 0.98, 0.18, 0.02])  # Adjust the position and size as needed

        # plt.tight_layout() 
        # append NN weithts here

        if(frame == 0):
            weights_list = []
            for network in ca_nn_list:
                state_dict = network.state_dict()
                flattened_params = []
                for param in state_dict.values():
                    flattened_params.extend(param.view(-1).tolist())
                weights_list.append(flattened_params)
            everystep_weights.append(weights_list)
            ca_grid = ca_grid
            grid_data = ca_grid[layer].cpu().numpy()
            if(enable_annotations_on_nca):
                cell_std = torch.std(ca_grid[0]) # Std applied on alpha channel only
                std_deviation_list = []
                for weights in weights_list:
                    # Convert the nested list to a PyTorch tensor
                    weights_tensor = torch.tensor(weights, device=DEVICE, dtype=torch.float32)
                    # Calculate the standard deviation for each nested list
                    std_deviation = torch.std(weights_tensor)
                    # Append the result to the list
                    std_deviation_list.append(std_deviation.item())
                std_deviation_tensor = torch.tensor(std_deviation_list, device=DEVICE, dtype=torch.float32)
                ann_std = torch.std(std_deviation_tensor)
                pixel_count = torch.nonzero(ca_grid[0]).size(0)
                num_all_zeros = sum(torch.all(torch.tensor(weights, device=DEVICE, dtype=torch.float32) == 0).item() for weights in weights_list)
                ann_count = (WIDTH*HEIGHT) - num_all_zeros
                plt.suptitle(f'Generation#{frame + 1}, AliveCells#{pixel_count}, AliveANNs#{ann_count}, Cell σ {round(cell_std.item(),3)}, ANNs σ  {round(ann_std.item(),3)}')
            else:
                plt.suptitle(f'Generation {frame + 1}')            
            min_value = 0
            max_value = 1
            norm = Normalize(vmin=min_value, vmax=max_value)
            for layer in range(NUM_LAYERS):
                ax = axes[layer]
                ax.clear()
                im = ax.imshow(ca_grid[layer].cpu().numpy(), cmap=colormaps[0],interpolation='none', norm=norm)
                ax.set_title(f'Layer {layer + 1}')
            # plt.title(f'Generation {frame + 1}')
            colorbar = fig.colorbar(im, cax=cax, orientation='horizontal', shrink=0.7)
            mid_value = (min_value + max_value) / 2
            ticks = [min_value, (min_value + mid_value) / 2, mid_value, (mid_value + max_value) / 2, max_value]  # Include midpoints
            colorbar.set_ticks(ticks)
            if(NUM_LAYERS==2):
                colorbar.ax.tick_params(axis='x', labelsize=5)
            elif(NUM_LAYERS==3):
                colorbar.ax.tick_params(axis='x', labelsize=6)
            else:
                colorbar.ax.tick_params(axis='x', labelsize=7)
            plt.subplots_adjust(top=0.9)
            plt.savefig(os.path.join('sim_frames_pdf', f'{frame:07d}.pdf'),format='pdf', dpi=600)
            plt.savefig(os.path.join('sim_frames_png', f'{frame:07d}.png'),format='png', dpi=600)
            writer.grab_frame()
        else:
            grid_data = ca_grid[layer].cpu().numpy()
            min_value = 0
            max_value = 1
            norm = Normalize(vmin=min_value, vmax=max_value)
            print("------------------------------------------------------------------")
            print(">>>>>>>>Simulation # {}".format(frame))
            ca_grid, ca_nn_list_updated_main = update_ca(ca_grid, ca_nn_list, frame)
            ca_grids_for_later_analysis.append(ca_grid[0].cpu().numpy())
            precision_multiplier = 10 ** precision
            rounded_grid = (ca_grid[0] * precision_multiplier).round() / precision_multiplier # picking only ALPHA values for plot!!!!!!
            unique_values, value_counts = torch.unique(rounded_grid, return_counts=True)
            frequency_dict = {value.item(): count.item() for value, count in zip(unique_values, value_counts)}
            frequency_dicts.append(frequency_dict)
            weights_list = []
            for network in ca_nn_list_updated_main:
                state_dict = network.state_dict()
                flattened_params = []
                for param in state_dict.values():
                    flattened_params.extend(param.view(-1).tolist())
                weights_list.append(flattened_params)
            everystep_weights.append(weights_list)
            if(enable_annotations_on_nca):
                cell_std = torch.std(ca_grid[0]) # Std applied on alpha channel only
                std_deviation_list = []
                for weights in weights_list:
                    # Convert the nested list to a PyTorch tensor
                    weights_tensor = torch.tensor(weights, device=DEVICE, dtype=torch.float32)
                    # Calculate the standard deviation for each nested list
                    std_deviation = torch.std(weights_tensor)
                    # Append the result to the list
                    std_deviation_list.append(std_deviation.item())
                std_deviation_tensor = torch.tensor(std_deviation_list, device=DEVICE, dtype=torch.float32)
                ann_std = torch.std(std_deviation_tensor)
                pixel_count = torch.nonzero(ca_grid[0]).size(0)
                num_all_zeros = sum(torch.all(torch.tensor(weights, device=DEVICE, dtype=torch.float32) == 0).item() for weights in weights_list)
                ann_count = (WIDTH*HEIGHT) - num_all_zeros
                plt.suptitle(f'Generation#{frame + 1}, AliveCells#{pixel_count}, AliveANNs#{ann_count}, Cell σ {round(cell_std.item(),3)}, ANNs σ  {round(ann_std.item(),3)}')
            else:
                plt.suptitle(f'Generation {frame + 1}')    
            for layer in range(NUM_LAYERS):
                ax = axes[layer]
                ax.clear()
                im = ax.imshow(ca_grid[layer].cpu().numpy(), cmap=colormaps[0],interpolation='none', norm=norm)
                ax.set_title(f'Layer {layer + 1}')
            # plt.title(f'Generation {frame + 1}')
            colorbar = fig.colorbar(im, cax=cax, orientation='horizontal', shrink=0.7)
            mid_value = (min_value + max_value) / 2
            ticks = [min_value, (min_value + mid_value) / 2, mid_value, (mid_value + max_value) / 2, max_value]  # Include midpoints
            colorbar.set_ticks(ticks)
            if(NUM_LAYERS==2):
                colorbar.ax.tick_params(axis='x', labelsize=5)
            elif(NUM_LAYERS==3):
                colorbar.ax.tick_params(axis='x', labelsize=6)
            else:
                colorbar.ax.tick_params(axis='x', labelsize=7)
            plt.subplots_adjust(top=0.9)
            plt.savefig(os.path.join('sim_frames_pdf', f'{frame:07d}.pdf'),format='pdf', dpi=600)
            plt.savefig(os.path.join('sim_frames_png', f'{frame:07d}.png'),format='png', dpi=600)
            writer.grab_frame()
            ca_nn_list = copy.deepcopy(ca_nn_list_updated_main)
        # fig.close()
        plt.close()
        plt.close(fig)
plt.close()
print("Simulation completed.")

import subprocess

nca_video_filename = "NCA_video_{}.mp4".format(stamp)
command = f"rm {nca_video_filename}"
subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


nca_video_filename = "NCA_video_{}".format(stamp)
# Save video as well

# Your Python variable for FPS and bitrate
fps = FPS  # replace with your desired value
bitrate = 10000  # replace with your desired value

# Construct the bash command with both FPS and bitrate variables
command = f"ffmpeg -framerate {fps} -pattern_type glob -i 'sim_frames_png/*.png' -c:v libx264 -b:v {bitrate}k -pix_fmt yuv420p {nca_video_filename}.mp4"

# Run the command quietly (suppress output)
subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Display the combined NCA Simulation
frames_folder = 'sim_frames_png'
frame_files = [f for f in os.listdir(frames_folder) if f.endswith(".png")]
frame_files.sort(key=lambda x: int(x.split(".")[0]))
frames = []
for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    frame = Image.open(frame_path)
    frames.append(frame)

# Define GIF-related parameters
output_gif_path = "NCA_gif_{}.gif".format(str(stamp))
desired_fps = FPS  # Add FPS definition
duration = int(1000 / desired_fps)

# Save frames as an animated GIF
frames[0].save(
    output_gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
    disposal=2,
    optimize=False
)

# Cleaning Current Directory
source_path = "NCA_gif_{}.gif".format(str(stamp))
destination_path = 'NCA'
shutil.move(source_path, destination_path)

source_path = "NCA_video_{}.mp4".format(stamp)
destination_path = 'NCA'
shutil.move(source_path, destination_path)

source_path = "sim_frames_png"
destination_path = 'NCA'
shutil.move(source_path, destination_path)

source_path = "sim_frames_pdf"
destination_path = 'NCA'
shutil.move(source_path, destination_path)







for step, frequency_dict in enumerate(frequency_dicts):
    print(f"Step {step + 1}:")
    for value, count in frequency_dict.items():
        print(f"  Value : {value}, Frequency : {count}")

for step, frequency_dict in enumerate(frequency_dicts):
    print(f"Step {step + 1}:")
    for value, count in list(frequency_dict.items()):
        print("before round off")
        print(f"  Value : {value}, Frequency : {count}")

        # Round off the 'value' to 1 decimal place and update the dictionary
        rounded_value = round(value, 1)
        frequency_dict[rounded_value] = frequency_dict.pop(value)

        print("after round off")
        print(f"  Value : {rounded_value}, Frequency : {count}")
for step, frequency_dict in enumerate(frequency_dicts):
    print(f"Step {step + 1}:")
    for value, count in frequency_dict.items():
        print(f"  Value : {value}, Frequency : {count}")

# Tool 1 PD
import matplotlib.pyplot as plt

# Set a larger figure size
plt.figure(figsize=(10, 6))  # You can adjust the width and height as needed

unique_keys = set()
for frequency_dict in frequency_dicts:
    unique_keys.update(frequency_dict.keys())

# Sort the unique_keys in ascending order
unique_keys = sorted(unique_keys)

x_values = list(range(0, num_steps))
y_values = {key: [] for key in unique_keys}
for step, frequency_dict in enumerate(frequency_dicts):
    for key in unique_keys:
        if key in frequency_dict:
            y_values[key].append(frequency_dict[key])
        else:
            y_values[key].append(0)

for key in unique_keys:
    plt.plot(x_values, y_values[key], label=f'Value {key:.{precision}f}')

plt.xlabel('Step Number')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("tool1_PD.pdf",format='pdf', dpi=600)
plt.show()
plt.close()
source_path = "tool1_PD.pdf"
destination_path = 'PD'
shutil.move(source_path, destination_path)
# Tool 2 PD

import math
Global_Entropies_H_ts = []
for step, frequency_dict in enumerate(frequency_dicts):
  total_count_W = 0
  ρ = []
  contributions = []
  Σ = 0
  H_t = 0
  total_states = 0
  for value, count in frequency_dict.items():
    if(value==0):
      total_states = 1
    else:
      total_states = total_states + 1
  for value, count in frequency_dict.items():
    total_count_W = total_count_W + count
  for value, count in frequency_dict.items():
    ρ.append(count/total_count_W)
  for each_ρ in ρ:
    contributions.append((-each_ρ) * math.log(each_ρ))
  for each in contributions:
    Σ = Σ + each
  if(math.log(total_states) == 0):
    H_t = Σ
  else:
    H_t = (1/math.log(total_states)) * Σ
  Global_Entropies_H_ts.append(H_t)

import matplotlib.pyplot as plt
Global_Entropies_H_ts = Global_Entropies_H_ts
steps = list(range(len(Global_Entropies_H_ts)))
plt.plot(steps, Global_Entropies_H_ts, marker='o', markersize=marker_size, linestyle='-')
plt.xlabel('Step')
plt.ylabel('Entropy (H_t)')
plt.title('Global Entropies over Steps')
plt.grid(True)
plt.savefig("tool2_PD.pdf",format='pdf', dpi=600)
plt.show()
plt.close()
source_path = "tool2_PD.pdf"
destination_path = 'PD'
shutil.move(source_path, destination_path)
# Tool 3 PD
medians = []
for step, frequency_dict in enumerate(frequency_dicts):
    print(f"Step {step + 1}:")
    values_with_repetitions = []
    for value, count in frequency_dict.items():
        values_with_repetitions.extend([value] * count)
    values_with_repetitions.sort()
    n = len(values_with_repetitions)
    if n % 2 == 0:
        median = (values_with_repetitions[n // 2 - 1] + values_with_repetitions[n // 2]) / 2
    else:
        median = values_with_repetitions[n // 2]
    medians.append(median)
    print(f"  Median: {median:.{precision}f}")


def round_list_elements(input_list, precision):
    rounded_list = [round(element, precision) for element in input_list]
    return rounded_list
medians = round_list_elements(medians, precision)

total_count_W = 0
for value, count in frequency_dict.items():
  total_count_W = total_count_W + count # Which is also equal to width * height


def kroncker_delta(x,y):
  if(x==y):
    return 1
  else:
    return 0
import math
ca_grids = ca_grids_for_later_analysis
gross_cell_variance = []
for y in range(len(ca_grids)): # total number_of_steps number of grids are there
  σ_gross = 0
  ca_grids_temp = ca_grids[y].flatten()
  for z in range(len(ca_grids_temp)):
    # print(round(ca_grids_temp[z],precision))
    δ = kroncker_delta(round(ca_grids_temp[z],precision),medians[y])
    σ_gross = σ_gross + (1-δ)
  gross_cell_variance.append(1/math.sqrt(total_count_W) * math.sqrt(σ_gross))

import matplotlib.pyplot as plt
gross_cell_variance = gross_cell_variance
steps = list(range(len(gross_cell_variance)))
plt.plot(steps, gross_cell_variance, marker='o', markersize=marker_size, linestyle='-')
plt.xlabel('Step')
plt.ylabel('Variance (σ_gross)')
plt.title('σ gross over Steps')
plt.grid(True)
plt.savefig("tool3_PD.pdf",format='pdf')
plt.show()
plt.close()
source_path = "tool3_PD.pdf"
destination_path = 'PD'
shutil.move(source_path, destination_path)
# Tool 4 PD
ca_grids = ca_grids_for_later_analysis
for each_grid in ca_grids:
  width = each_grid.shape[0]
  height = each_grid.shape[1]
  for i in range(width):
    for j in range(height):
      cell_value = each_grid[i][j]
      # print("value before",cell_value)
      updated = round(cell_value,precision)
      each_grid[i][j] = updated
      # print("value after",updated)

print(ca_grids)

def round_dict_keys(input_dict, precision):
    rounded_dict = {}
    for key, value in input_dict.items():
        rounded_key = round(key, precision)
        rounded_dict[rounded_key] = value
    return rounded_dict

normalized_frequency_dicts = []
for step, frequency_dict in enumerate(frequency_dicts):
    min_count = min(frequency_dict.values())
    max_count = max(frequency_dict.values())
    normalized_frequency_dict = {}
    for value, count in frequency_dict.items():
        if(min_count == max_count):
          normalized_count = 0
          normalized_frequency_dict[value] = normalized_count
        else:
          normalized_count = (count - min_count) / (max_count - min_count)
          normalized_frequency_dict[value] = normalized_count
    normalized_frequency_dicts.append(round_dict_keys(normalized_frequency_dict,precision))
for step, normalized_frequency_dict in enumerate(normalized_frequency_dicts):
    print(f"Step {step + 1}:")
    for value, normalized_count in normalized_frequency_dict.items():
        print(f"  Value {value:}: Normalized Frequency {normalized_count}")

def get_neighbors_with_wrap_around(grid, row, col, window_size, grid_size):
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            new_row, new_col = (row + i) % grid_size, (col + j) % grid_size
            neighbors.append(grid[new_row][new_col])
    return neighbors

window_size = 1

def frequency_of(key, my_dict):
    # print("key as input", key)
    key = round(key,precision)
    key = min(my_dict, key=lambda x: abs(x - key))
    # print("key after precision", key)
    if key in my_dict:
        return my_dict[key]
    else:
        return None
import copy
mu_loc_data = []
for each_grid,frequency_dict_temp in zip(ca_grids,normalized_frequency_dicts):
  # frequency_dict_temp = round_dict_keys(frequency_dict_temp,precision)
  width = each_grid.shape[0]
  height = each_grid.shape[1]
  grid_size = each_grid.shape[0]
  rows = width # Replace 'rows' with the number of rows you want
  cols = height  # Replace 'cols' with the number of columns you want
  mu_locs_list_2d = []
  for i in range(rows):
      mu_locs_list_2d.append([None] * cols)
  for i in range(width):
    for j in range(height):
      current_cell = each_grid[i][j]
      current_cell_index_row = i
      current_cell_index_column = j
      # print(each_grid)
      neighbors_of_current_cell = get_neighbors_with_wrap_around(each_grid,current_cell_index_row,current_cell_index_column,window_size,grid_size)
      # print("neighbors",neighbors_of_current_cell)
      sum_of_frequencies_of_cell = 0
      for each_neighbor in neighbors_of_current_cell:
        # print("pciked neighbor",each_neighbor)
        # print("frequency dict",frequency_dict_temp)
        sum_of_frequencies_of_cell = sum_of_frequencies_of_cell + frequency_of(each_neighbor,frequency_dict_temp)
      mu_loc_current_cell = sum_of_frequencies_of_cell/9
      mu_locs_list_2d[i][j] = mu_loc_current_cell
  mu_loc_data.append(mu_locs_list_2d)



def get_neighbors_with_wrap_around(grid, row, col, window_size, grid_size):
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            new_row, new_col = (row + i) % grid_size, (col + j) % grid_size
            neighbors.append(grid[new_row][new_col])
    return neighbors

window_size = 1

def frequency_of(key, my_dict):
    # print("key as input", key)
    key = round(key,precision)
    key = min(my_dict, key=lambda x: abs(x - key))
    # print("key after precision", key)
    if key in my_dict:
        return my_dict[key]
    else:
        return None
import copy
import math
sigma_loc_data = []
for each_grid,frequency_dict_temp,mu_data in zip(ca_grids,normalized_frequency_dicts,mu_loc_data):
  # frequency_dict_temp = round_dict_keys(frequency_dict_temp,precision)
  width = each_grid.shape[0]
  height = each_grid.shape[1]
  grid_size = each_grid.shape[0]
  rows = width
  cols = height
  sigma_locs_list_2d = []
  for i in range(rows):
      sigma_locs_list_2d.append([None] * cols)
  for i in range(width):
    for j in range(height):
      current_cell = each_grid[i][j]
      current_cell_index_row = i
      current_cell_index_column = j
      neighbors_of_current_cell = get_neighbors_with_wrap_around(each_grid,current_cell_index_row,current_cell_index_column,window_size,grid_size)
      sum_of_squared_frequencies = 0
      for each_neighbor in neighbors_of_current_cell:
        difference = frequency_of(each_neighbor,frequency_dict_temp) - mu_data[i][j]
        squared = difference ** 2
        sum_of_squared_frequencies = sum_of_squared_frequencies + squared
      sigma_loc_current_cell = math.sqrt(sum_of_squared_frequencies)/3
      sigma_locs_list_2d[i][j] = sigma_loc_current_cell
  sigma_loc_data.append(sigma_locs_list_2d)



mu_global = []
for a in range(len(ca_grids)):
  mu_g = 0
  W = ca_grids[a].shape[0] * ca_grids[a].shape[1]
  for x in range(ca_grids[a].shape[0]):
    for y in range(ca_grids[a].shape[1]):
      mu_g = mu_g + sigma_loc_data[a][x][y]
  mu_g = mu_g/W
  mu_global.append(mu_g)

# print(mu_global)

sigma_global = []
for a in range(len(ca_grids)):
  mu_glob = mu_global[a]
  sigma_sum = 0
  W = ca_grids[a].shape[0] * ca_grids[a].shape[1]
  for x in range(ca_grids[a].shape[0]):
    for y in range(ca_grids[a].shape[1]):
      sigma_sum = sigma_sum + (sigma_loc_data[a][x][y] - mu_glob)**2
  sigma_root = math.sqrt(sigma_sum)
  sigma_globa = sigma_root/math.sqrt(W)
  sigma_global.append(sigma_globa)
# print(sigma_global)
  

import matplotlib.pyplot as plt
sigma_global = sigma_global
steps = list(range(len(sigma_global)))
plt.plot(steps, sigma_global, marker='o', markersize=marker_size, linestyle='-')
plt.xlabel('Step')
plt.ylabel('Variance (σ_global)')
plt.title('σ global over Steps using Local Organzation of Cells')
plt.grid(True)
plt.savefig("tool4_PD.pdf",format='pdf', dpi=600)
plt.show()
plt.close()
source_path = "tool4_PD.pdf"
destination_path = 'PD'
shutil.move(source_path, destination_path)
# Plot tools PD 2,3,4
import matplotlib.pyplot as plt
plot_data_tool2 = Global_Entropies_H_ts
plot_data_tool3 = gross_cell_variance
plot_data_tool4 = sigma_global
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(plot_data_tool2, marker='o', markersize=marker_size, label='H$_t$') #  for Tool 2
ax.plot(plot_data_tool3, marker='s', markersize=marker_size, label='σ$_{gross}$') # for Tool 3
ax.plot(plot_data_tool4, marker='^', markersize=marker_size, label='σ$_{glob}$') # for Tool 4
ax.set_xlabel('X-axis Label')
ax.set_ylabel('Y-axis Label')
ax.set_title('All Tool Information in a Single Plot')
ax.legend()
plt.grid(True)
plt.savefig("tool234_PD.pdf",format='pdf', dpi=600)
plt.show()
plt.close()
source_path = "tool234_PD.pdf"
destination_path = 'PD'
shutil.move(source_path, destination_path)

# Plot tools PD 2,4
import matplotlib.pyplot as plt
plot_data_tool2 = Global_Entropies_H_ts
plot_data_tool4 = sigma_global
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(plot_data_tool2, marker='o', markersize=marker_size, label='H$_t$') #  for Tool 2
# ax.plot(plot_data_tool3, marker='s', markersize=marker_size, label='σ$_{gross}$') # for Tool 3
ax.plot(plot_data_tool4, marker='^', markersize=marker_size, label='σ$_{glob}$') # for Tool 4
ax.set_xlabel('X-axis Label')
ax.set_ylabel('Y-axis Label')
ax.set_title('All Tool Information in a Single Plot')
ax.legend()
plt.grid(True)
plt.savefig("tool24_PD.pdf",format='pdf', dpi=600)
plt.show()
plt.close()
source_path = "tool24_PD.pdf"
destination_path = 'PD'
shutil.move(source_path, destination_path)


# GD Tools:
# RWSP Tool Tool 1
import numpy as np
import random
import math
import matplotlib.pyplot as plt
number_of_random_weights = 3

import math
this_list_should_contain_NN_for_every_step = everystep_weights
def round_elements_in_nested_list(input_list, precision=2):
    if isinstance(input_list, list):
        return [round_elements_in_nested_list(elem, precision) for elem in input_list]
    else:
        return round(input_list, precision)

rounded_list = round_elements_in_nested_list(this_list_should_contain_NN_for_every_step, precision)
rounded_list_further = rounded_list
round_list_array = np.array(rounded_list_further)
steps_value = round_list_array.shape[0]
row_col = round_list_array.shape[1]
params = round_list_array.shape[2]
rounded_genes_final = round_list_array.reshape(steps_value,int(math.sqrt(row_col)),int(math.sqrt(row_col)),params)

rounded_array_further = np.zeros((steps_value,int(math.sqrt(row_col)),int(math.sqrt(row_col)),number_of_random_weights))
random_postions = random.sample(range(params), number_of_random_weights)
for i in range(steps_value):
    for j in range(int(math.sqrt(row_col))):
        for k in range(int(math.sqrt(row_col))):
            selected_values_array = rounded_genes_final[i][j][k]
            selected_values = np.array([selected_values_array[i] for i in random_postions])
            rounded_array_further[i][j][k] = selected_values
import numpy as np
import matplotlib.pyplot as plt


data = rounded_array_further
normalized_data = (data * 255).astype(np.uint8)

length = len(normalized_data)

if not os.path.exists('gd_rwsp_frames_png'):
    os.makedirs('gd_rwsp_frames_png')
if not os.path.exists('gd_rwsp_frames_pdf'):
    os.makedirs('gd_rwsp_frames_pdf')

def save_frame(frame, fig, normalized_data, counts_per_frame):
    im = plt.imshow(normalized_data[frame], cmap='jet')
    plt.title(f'Generation {frame + 1}')
    # cax = fig.add_axes([0.08, 0.94, 0.15, 0.02])
    # colorbar = fig.colorbar(im, cax=cax, orientation='horizontal', shrink=0.7)
    # min_value = np.min(normalized_data[frame])
    # max_value = np.max(normalized_data[frame])
    # mid_value = (min_value + max_value) / 2
    # ticks = [min_value, (min_value + mid_value) / 2, mid_value, (mid_value + max_value) / 2, max_value]  # Include midpoints
    # rounded_ticks = [round(value) for value in ticks]
    # ticks = rounded_ticks
    # colorbar.set_ticks(ticks)
    # colorbar.ax.tick_params(axis='x', labelsize=6)

    # Count unique RGB colors
    flattened_data = normalized_data[frame].reshape(-1, 3)
    print("flattened_data")
    print(flattened_data)
    unique_colors, counts = np.unique(flattened_data, axis=0, return_counts=True)
    print("unique_colors")
    print(unique_colors)    
    unique_colors_count = len(unique_colors)

    # Store the count of unique RGB colors for this frame
    counts_per_frame.append(unique_colors_count-1)

    # Save the plot
    plt.savefig(os.path.join('gd_rwsp_frames_png', f"{frame + 1:07d}.png"), format='png', dpi=600)
    plt.savefig(os.path.join('gd_rwsp_frames_pdf', f"{frame + 1:07d}.pdf"), format='pdf', dpi=600)
    plt.clf()

# Example usage
counts_per_frame = []

for frame in range(length):
    fig = plt.figure()
    save_frame(frame, fig, normalized_data, counts_per_frame)
    plt.close(fig)


counts_rwsp = counts_per_frame
# Plot the count of unique RGB colors
plt.figure(figsize=(12, 8))
plt.plot(range(0, length), counts_per_frame, marker='o', linestyle='-', label='Unique RGB Colors Count - RWSP', color='blue')
plt.xlabel('Generation')
plt.ylabel('Count')
plt.title('Unique RGB Colors Count per Generation - RWSP')
plt.legend()
plt.savefig('unique_rgb_colors_count_plot_rwsp.png', format='png', dpi=600)
plt.savefig('unique_rgb_colors_count_plot_rwsp.pdf', format='pdf', dpi=600)
plt.show()



# Display the combined clustering plot
frames_folder = 'gd_rwsp_frames_png'
frame_files = [f for f in os.listdir(frames_folder) if f.endswith(".png")]
frame_files.sort(key=lambda x: int(x.split(".")[0]))
frames = []
for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    frame = Image.open(frame_path)
    frames.append(frame)

# Define GIF-related parameters
output_gif_path = "tool1_gd_rwsp_gif.gif"
desired_fps = FPS  # Add FPS definition
duration = int(1000 / desired_fps)

# Save frames as an animated GIF
frames[0].save(
    output_gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
    disposal=2,
    optimize=False
)

# Save video as well
import subprocess

# Your Python variable for FPS and bitrate
fps = FPS  # replace with your desired value
bitrate = 10000  # replace with your desired value

# Construct the bash command with both FPS and bitrate variables
command = f"ffmpeg -framerate {fps} -pattern_type glob -i 'gd_rwsp_frames_png/*.png' -c:v libx264 -b:v {bitrate}k -pix_fmt yuv420p tool1_gd_rwsp_video.mp4"

# Run the command quietly (suppress output)
subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

source_path = "tool1_gd_rwsp_video.mp4"
destination_path = 'GD'
shutil.move(source_path, destination_path)

source_path = "tool1_gd_rwsp_gif.gif"
destination_path = 'GD'
shutil.move(source_path, destination_path)

source_path = "gd_rwsp_frames_png"
destination_path = 'GD'
shutil.move(source_path, destination_path)

source_path = "gd_rwsp_frames_pdf"
destination_path = 'GD'
shutil.move(source_path, destination_path)


source_path = "unique_rgb_colors_count_plot_rwsp.png"
destination_path = 'GD'
shutil.move(source_path, destination_path)


source_path = "unique_rgb_colors_count_plot_rwsp.pdf"
destination_path = 'GD'
shutil.move(source_path, destination_path)

# GHC Tool Tool 2
import numpy as np
import random

list_of_weights_at_every_step = round_elements_in_nested_list(this_list_should_contain_NN_for_every_step)
sample = list_of_weights_at_every_step
sample = np.array(sample)
params = sample[0][0].shape[0]
print("PARAMSPARAMSPARAMSPARAMSPARAMSPARAMSPARAMSPARAMS")
print(params)
print("PARAMSPARAMSPARAMSPARAMSPARAMSPARAMSPARAMSPARAMS")
sample = sample.reshape(NUM_STEPS+1,WIDTH,HEIGHT,params)
print(sample.shape)
height, width = sample.shape[1], sample.shape[2]
length_sim = sample.shape[0]

import random
import numpy as np
import matplotlib.pyplot as plt
import os
output_folder = "gd_ghc_frames_png"
output_folder2 = "gd_ghc_frames_pdf"

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)
# sample[8][0] # weights of the NN at different time steps
# fig, ax = plt.subplots()
ghc_scaling_factor = 10
counts_per_frame = []
for i in range(length_sim):
  # Function to hash a 44-bit gene sequence to 24 bits using Python's built-in hash function
  def hash_gene_sequence(gene_sequence):
      return [int(hash(value) % 2) for value in gene_sequence]

  # Function to convert a 24-bit sequence to three 8-bit values
  def split_to_rgb(sequence):
      r = int(''.join(map(str, sequence[:8])), 2)
      g = int(''.join(map(str, sequence[8:16])), 2)
      b = int(''.join(map(str, sequence[16:24])), 2)
      return r, g, b

  # Create an empty grid to store the RGB values for each cell
  grid = np.zeros((height, width, 3), dtype=np.uint8)

  # Populate the grid with color-coded gene sequences
  for row in range(height):
      for col in range(width):
          gene_sequence = ghc_scaling_factor * sample[i][row][col]
          gene_sequence = gene_sequence.tolist()
          hashed_sequence = hash_gene_sequence(gene_sequence)
          r, g, b = split_to_rgb(hashed_sequence)
          grid[row, col] = [r, g, b]
  frame_filename = os.path.join(output_folder2, f"{i + 1:07d}.pdf")
  fig = plt.figure()
  im = plt.imshow(grid,interpolation='none',cmap='jet')
  # Count unique RGB colors
  flattened_data = grid.reshape(-1, 3)
  print("flattened_data")
  print(flattened_data)
  unique_colors, counts = np.unique(flattened_data, axis=0, return_counts=True)
  print("unique_colors")
  print(unique_colors)    
  unique_colors_count = len(unique_colors)
  # Store the count of unique RGB colors for this frame
  counts_per_frame.append(unique_colors_count-1)  
  plt.title(f'Generation {i + 1}')
#   cax = fig.add_axes([0.08, 0.94, 0.15, 0.02])
#   colorbar = fig.colorbar(im, cax=cax, orientation='horizontal', shrink=0.7)
#   min_value = np.min(grid)
#   max_value = np.max(grid)
#   mid_value = (min_value + max_value) / 2
#   ticks = [min_value, (min_value + mid_value) / 2, mid_value, (mid_value + max_value) / 2, max_value]  # Include midpoints
#   rounded_ticks = [round(value) for value in ticks]
#   ticks = rounded_ticks
#   colorbar.set_ticks(ticks)
#   colorbar.ax.tick_params(axis='x', labelsize=6)
  # plt.axis('off')
  plt.savefig(frame_filename,format='pdf',dpi=600)
  frame_filename = os.path.join(output_folder, f"{i + 1:07d}.png")
  plt.savefig(frame_filename,format='png',dpi=600)
  plt.close()


counts_ghc = counts_per_frame
# Plot the count of unique RGB HASH colors
plt.figure(figsize=(12, 8))
plt.plot(range(0, length_sim), counts_per_frame, marker='o', linestyle='-', label='Unique RGB Colors Count - GHC', color='blue')
plt.xlabel('Generation')
plt.ylabel('Count')
plt.title('Unique RGB Colors Count per Generation - GHC')
plt.legend()
plt.savefig('unique_rgb_colors_count_plot_ghc.png', format='png', dpi=600)
plt.savefig('unique_rgb_colors_count_plot_ghc.pdf', format='pdf', dpi=600)
plt.show()


# Display the combined clustering plot
frames_folder = output_folder
frame_files = [f for f in os.listdir(frames_folder) if f.endswith(".png")]
frame_files.sort(key=lambda x: int(x.split(".")[0]))
frames = []
for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    frame = Image.open(frame_path)
    frames.append(frame)

# Define GIF-related parameters
output_gif_path = "tool2_gd_ghc_gif.gif"
desired_fps = FPS  # Add FPS definition
duration = int(1000 / desired_fps)

# Save frames as an animated GIF
frames[0].save(
    output_gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
    disposal=2,
    optimize=False
)


# Save video as well
import subprocess

# Your Python variable for FPS and bitrate
fps = FPS  # replace with your desired value
bitrate = 10000  # replace with your desired value

# Construct the bash command with both FPS and bitrate variables
command = f"ffmpeg -framerate {fps} -pattern_type glob -i 'gd_ghc_frames_png/*.png' -c:v libx264 -b:v {bitrate}k -pix_fmt yuv420p tool2_gd_ghc_video.mp4"

# Run the command quietly (suppress output)
subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)




import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
generations = range(0, length_sim)
# Plot counts_rwsp with a solid line and square markers
plt.plot(generations, counts_rwsp, linestyle='-', marker='s', color='blue', label='RWSP', markersize=0.5*marker_size)

# Plot counts_ghc with a dashed line and circle markers
plt.plot(generations, counts_ghc, linestyle='-', marker='o', color='red', label='GHC', markersize=0.5*marker_size)

plt.xlabel('Generation')
plt.ylabel('Unique Colors')
plt.title('Unqiue Color Counts for RWSP and GHC Plot')
plt.legend()
# plt.grid(True)  # Add grid for better readability

# Calculate step size automatically
step_size = max(1, len(generations) // 20)  # You can adjust 20 based on your preference
plt.xticks(range(0, length_sim, step_size))  # Adjust the range and step as needed

plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig('combined_rwsp_ghc.png', format='png', dpi=600)  # Save plot as PNG
plt.savefig('combined_rwsp_ghc.pdf', format='pdf', dpi=600)  # Save plot as PNG
plt.show()
plt.close()



source_path = "tool2_gd_ghc_video.mp4"
destination_path = 'GD'
shutil.move(source_path, destination_path)

source_path = "tool2_gd_ghc_gif.gif"
destination_path = 'GD'
shutil.move(source_path, destination_path)

source_path = "gd_ghc_frames_png"
destination_path = 'GD'
shutil.move(source_path, destination_path)

source_path = "gd_ghc_frames_pdf"
destination_path = 'GD'
shutil.move(source_path, destination_path)

source_path = "unique_rgb_colors_count_plot_ghc.png"
destination_path = 'GD'
shutil.move(source_path, destination_path)


source_path = "unique_rgb_colors_count_plot_ghc.pdf"
destination_path = 'GD'
shutil.move(source_path, destination_path)



source_path = "combined_rwsp_ghc.png"
destination_path = 'GD'
shutil.move(source_path, destination_path)


source_path = "combined_rwsp_ghc.pdf"
destination_path = 'GD'
shutil.move(source_path, destination_path)


# Outputs

source_path = "GD"
destination_path = 'Outputs_'+str(output_stamp)
shutil.move(source_path, destination_path)

source_path = "NCA"
destination_path = 'Outputs_'+str(output_stamp)
shutil.move(source_path, destination_path)

source_path = "PD"
destination_path = 'Outputs_'+str(output_stamp)
shutil.move(source_path, destination_path)

source_path = "interestingoutput.out"
destination_path = 'Outputs_'+str(output_stamp)
shutil.move(source_path, destination_path)
