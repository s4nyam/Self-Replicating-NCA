precision = 1
WIDTH, HEIGHT = 200,200
grid_size = (WIDTH, HEIGHT)
INIT_PROBABILITY = 0.05
min_pixels = max(0, int(WIDTH * HEIGHT * INIT_PROBABILITY))
NUM_LAYERS = 2 # One hidden and one alpha
ALPHA = 0.6 # To make other cells active
INHERTIANCE_PROBABILITY  = 0.1 # probability that neighboring cells will inherit by perturbation.
parameter_perturbation_probability = 0.05
NUM_STEPS = 1000
num_steps = NUM_STEPS
activation = 'sigmoid' # ['relu','sigmoid','tanh','leakyrelu']
FPS = 1 # Speed of display for animation of NCA and plots
marker_size = 2 # for plots


frequency_dicts = []
everystep_weights = [] 
ca_grids_for_later_analysis_layer0 = []
ca_grids_for_later_analysis_layer1 = []

import pickle
import torch
torch.set_printoptions(precision=precision)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import random
import copy
from collections import Counter
import os
frame_folder = 'frames'
os.makedirs(frame_folder, exist_ok=True)
# Define a function to save a frame with a formatted file name
def save_frame(ca_grid, frame_number):
    for layer in range(NUM_LAYERS):
        plt.imshow(ca_grid[layer].cpu().numpy(), cmap=colormaps[layer])
        plt.title(f'Layer {layer + 1}')
        fig = plt.gcf()
        fig.set_dpi(800)
        plt.savefig(os.path.join(frame_folder, f'frame_{frame_number:07d}_layer_{layer}.pdf'), format='pdf', dpi=fig.get_dpi())
        plt.close()


ca_grids_for_later_analysis = []

# def custom_activation(x):
#     result = 0.05 * (x + 2)
#     result = max(0, min(0.1, result))
#     return result
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
output_size_fc1 = 2
weight_params_fc1 = input_size_fc1 * output_size_fc1
bias_params_fc1 = output_size_fc1

input_size_fc2 = 2
output_size_fc2 = NUM_LAYERS
weight_params_fc2 = input_size_fc2 * output_size_fc2
bias_params_fc2 = output_size_fc2

weight_params = weight_params_fc1 + weight_params_fc2
bias_params = bias_params_fc1 + bias_params_fc2
total_params = weight_params + bias_params



ca_grid = torch.zeros((NUM_LAYERS, WIDTH, HEIGHT), device=DEVICE, dtype=torch.float32)
random_tensor = torch.ones((WIDTH, HEIGHT))
sorted_values, indices = random_tensor.view(-1).sort()
mask = torch.zeros_like(random_tensor)
shuffled_indices = indices.tolist()  # Convert indices to a list
random.shuffle(shuffled_indices)     # Shuffle the list
random_positions = shuffled_indices[:min_pixels]
mask.view(-1)[random_positions] = 1.0
ca_grid[0] = mask * (ALPHA + (1 - ALPHA) * random_tensor)  # Initialize the alpha channel with values greater than ALPHA
ca_grid[1] = mask * (ALPHA + (1 - ALPHA) * random_tensor * random.random())  # Initialize the other channel with values greater than ALPHA
# ca_grid[0] = mask * (random_tensor) # Initialising only alpha channel
# ca_grid[1] = mask * (random_tensor * random.random()) # Initialising a little amount of channel 1 with the idea that if a pixel is having some intensity in alpha, it should have some values in other channels.


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)
def initialize_weights_to_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        nn.init.zeros_(module.bias)

# Create a list of neural networks, one for each pixel
ca_nn_list = [SimpleNN().to(DEVICE) for _ in range(WIDTH * HEIGHT)]

# Initialize the weights for neural networks associated with live pixels
for i in range(WIDTH):
    for j in range(HEIGHT):
        if ca_grid[0, i, j] > ALPHA:
            ca_nn_list[i * WIDTH + j].apply(initialize_weights)


def update_ca(ca_grid, ca_nn):
    new_ca_grid = ca_grid.clone()
    def process_neighborhood(i, j, idx):
        neighborhood = torch.zeros(9 * NUM_LAYERS, device=DEVICE)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                ni, nj = (i + dx) % WIDTH, (j + dy) % HEIGHT
                for l in range(NUM_LAYERS):
                    neighborhood[(dx + 1) * 3 + (dy + 1) + l * 9] = ca_grid[l, ni, nj]
        neighborhood = torch.unsqueeze(neighborhood, 0)
        output = None  # Initialize output to None

        if((neighborhood > ALPHA).any()):
          output = ca_nn_list[idx](neighborhood)
          if(random.random() < INHERTIANCE_PROBABILITY):
            high_alpha_pixels = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if ca_grid[0, (i + dx) % WIDTH, (j + dy) % HEIGHT] > ALPHA] # Find neighboring pixels with alpha values greater than ALPHA
            if high_alpha_pixels:
                # check again for livelihood of the pixels (if this if condition passed, that means there is at least one element that is greater than alpha.)
                current_alpha_value = ca_grid[0, i, j]
                high_alpha_values = [ca_grid[0, (i + dx) % WIDTH, (j + dy) % HEIGHT] for dx, dy in high_alpha_pixels]
                selected_pixel = random.choice(high_alpha_pixels)  # Select any random live pixel in the neighborhood
                ni, nj = (i + selected_pixel[0]) % WIDTH, (j + selected_pixel[1]) % HEIGHT

                # Check if the selected neural network has any non-zero weights
                selected_nn_idx = ni * WIDTH + nj
                selected_nn = ca_nn_list[selected_nn_idx]
                has_nonzero_weights = any(torch.any(param != 0) for param in selected_nn.parameters())

                if not has_nonzero_weights:
                    # Copy the neural network from the selected neighboring pixel
                    ca_nn_list[idx] = copy.deepcopy(selected_nn)

                    # Perturb neural network weights
                    for name, param in ca_nn_list[idx].named_parameters():
                        if 'weight' in name:
                            if random.random() < parameter_perturbation_probability:
                                param.data += torch.randn_like(param.data) * random.uniform(-1, 1)
                else:
                    ca_nn_list[idx] = copy.deepcopy(selected_nn)
        else:
          ca_nn_list[idx].apply(initialize_weights_to_zero)


        if output is not None:
            return output.squeeze().tolist()
        else:
            return [0.0] * NUM_LAYERS

    idx = 0  # Index for the neural networks
    for i in range(WIDTH):
        for j in range(HEIGHT):
            updated_values = process_neighborhood(i, j, idx)
            for layer in range(NUM_LAYERS):
                new_ca_grid[layer, i, j] = updated_values[layer]
            idx += 1

    new_ca_grid_temp = new_ca_grid.clone()
    # now replace all pixels with their corresponding sigmoid values.
    for x in range(WIDTH):
      for y in range(HEIGHT):
        for layer in range(NUM_LAYERS):
          updated_value = sigmoid(new_ca_grid_temp[layer, x, y].cpu().numpy())
          new_ca_grid_temp[layer, x, y] = updated_value # setting a value between 0 and 1 for the alive pixels.

    # now we will check for the ALPHA values as threshold
    for x in range(WIDTH):
      for y in range(HEIGHT):
          # Check if any value in channel 0 is less than ALPHA at the current position
          if (new_ca_grid_temp[0, x, y] < ALPHA):
            # If any value is less than ALPHA, set values in all layers at the current position to 0
            for layer in range(NUM_LAYERS):
                new_ca_grid_temp[layer, x, y] = 0.0

    return new_ca_grid_temp




metadata = dict(title='Neural CA Simulation', artist='AI', comment='Neural CA Simulation')
writer = FFMpegWriter(fps=FPS, metadata=metadata)
all_colormaps = plt.colormaps()
colormaps = all_colormaps
fig, axes = plt.subplots(1, NUM_LAYERS, figsize=(5 * NUM_LAYERS, 5))
plt.tight_layout()
plt.close(fig)
import time
stamp = int(time.time())

for frame in range(NUM_STEPS+1):
    # append NN weithts here
    weights_list = []
    for network in ca_nn_list:
        state_dict = network.state_dict()
        flattened_params = []
        for param in state_dict.values():
            flattened_params.extend(param.view(-1).tolist())
        weights_list.append(flattened_params)
    everystep_weights.append(weights_list)
    if(frame == 0):
      ca_grid = ca_grid
      for layer in range(NUM_LAYERS):
          ax = axes[layer]
          ax.clear()
          ax.imshow(ca_grid[layer].cpu().numpy(), cmap=colormaps[layer])
          ax.set_title(f'Layer {layer + 1}')
      save_frame(ca_grid, frame)
    else:
      ca_grid = update_ca(ca_grid, ca_nn)
      ca_grids_for_later_analysis_layer0.append(ca_grid[0].cpu().numpy())
      ca_grids_for_later_analysis_layer1.append(ca_grid[1].cpu().numpy())
      precision_multiplier = 10 ** precision
      rounded_grid = (ca_grid[0] * precision_multiplier).round() / precision_multiplier # picking only ALPHA values for plot!!!!!!
      unique_values, value_counts = torch.unique(rounded_grid, return_counts=True)
      frequency_dict = {value.item(): count.item() for value, count in zip(unique_values, value_counts)}
      frequency_dicts.append(frequency_dict)
      for layer in range(NUM_LAYERS):
          ax = axes[layer]
          ax.clear()
          ax.imshow(ca_grid[layer].cpu().numpy(), cmap=colormaps[layer])
          ax.set_title(f'Layer {layer + 1}')
      save_frame(ca_grid, frame)
    if(frame%20==0 or frame<20):
      with open('frequency_dicts.pkl', 'wb') as f1, open('everystep_weights.pkl', 'wb') as f2, open('ca_grids_layer0.pkl', 'wb') as f3, open('ca_grids_layer1.pkl', 'wb') as f4:
        pickle.dump(frequency_dicts, f1)
        pickle.dump(everystep_weights, f2)
        pickle.dump(ca_grids_for_later_analysis_layer0, f3)
        pickle.dump(ca_grids_for_later_analysis_layer1, f4)



# Save these three lists to storage and know method to reload them
# frequency_dicts = []
# everystep_weights = [] 
# ca_grids_for_later_analysis_layer0 = []
# ca_grids_for_later_analysis_layer1 = []


# #Reload
# import pickle

# # Load the lists from the binary files
# with open('frequency_dicts.pkl', 'rb') as f1, open('everystep_weights.pkl', 'rb') as f2, open('ca_grids_layer0.pkl', 'rb') as f3, open('ca_grids_layer1.pkl', 'rb') as f4:
#     frequency_dicts = pickle.load(f1)
#     everystep_weights = pickle.load(f2)
#     ca_grids_for_later_analysis_layer0 = pickle.load(f3)
#     ca_grids_for_later_analysis_layer1 = pickle.load(f4)


# # Now, you have your lists back in memory
# print(frequency_dicts)
# print(everystep_weights)
# print(ca_grids_for_later_analysis_layer0)
# print(ca_grids_for_later_analysis_layer1)