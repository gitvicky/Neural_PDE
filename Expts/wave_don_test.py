# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DeepONetMultipleOutputs(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, branch_hidden_layers, trunk_hidden_layers, output_dim):
        super(DeepONetMultipleOutputs, self).__init__()
        
        self.branch_net = self._create_mlp(branch_input_dim, branch_hidden_layers, output_dim)
        self.trunk_net = self._create_mlp(trunk_input_dim, trunk_hidden_layers, output_dim)
        
    def _create_mlp(self, input_dim, hidden_layers, output_dim):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        
        return torch.sum(branch_output * trunk_output, dim=-1)

def generate_gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0)**2 / (2 * sigma_x**2) + (y - y0)**2 / (2 * sigma_y**2)))

def generate_dataset(num_simulations, grid_size, num_timesteps):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    t = np.linspace(0, 1, num_timesteps)
    X, Y = np.meshgrid(x, y)
    
    dataset = []
    
    for _ in range(num_simulations):
        amplitude = np.random.uniform(0.5, 2.0)
        x0, y0 = np.random.uniform(0, 1, 2)
        sigma_x, sigma_y = np.random.uniform(0.05, 0.2, 2)
        
        initial_condition = generate_gaussian_2d(X, Y, amplitude, x0, y0, sigma_x, sigma_y)
        
        # Simulate wave equation (simplified)
        u = np.zeros((num_timesteps, grid_size, grid_size))
        u[0] = initial_condition
        u[1] = initial_condition
        
        c = 1  # Wave speed
        dx = 1 / (grid_size - 1)
        dt = 1 / (num_timesteps - 1)
        
        for n in range(1, num_timesteps - 1):
            u[n+1][1:-1,1:-1] = 2*u[n][1:-1,1:-1] - u[n-1][1:-1,1:-1] + c**2 * dt**2 * (
                (u[n][2:, 1:-1] - 2*u[n][1:-1, 1:-1] + u[n][:-2, 1:-1]) / dx**2 +
                (u[n][1:-1, 2:] - 2*u[n][1:-1, 1:-1] + u[n][1:-1, :-2]) / dx**2
            )
        
        dataset.append((initial_condition.flatten(), u))
    
    return dataset

def train_deeponet(model, dataset, num_epochs, batch_size, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for initial_condition, u in dataset:
            branch_input = torch.FloatTensor(initial_condition).unsqueeze(0)
            
            for t in range(u.shape[0]):
                x = np.linspace(0, 1, u.shape[1])
                y = np.linspace(0, 1, u.shape[2])
                X, Y = np.meshgrid(x, y)
                trunk_input = torch.FloatTensor(np.column_stack((X.flatten(), Y.flatten(), np.full(X.size, t/u.shape[0]))))
                
                target = torch.FloatTensor(u[t].flatten())
                
                optimizer.zero_grad()
                output = model(branch_input, trunk_input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def visualize_results(model, initial_condition, grid_size, num_timesteps):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    branch_input = torch.FloatTensor(initial_condition).unsqueeze(0)
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, t in enumerate([0, num_timesteps//2, num_timesteps-1]):
        trunk_input = torch.FloatTensor(np.column_stack((X.flatten(), Y.flatten(), np.full(X.size, t/num_timesteps))))
        
        output = model(branch_input, trunk_input).detach().numpy().reshape(grid_size, grid_size)
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot_surface(X, Y, output, cmap='viridis')
        ax.set_title(f"t = {t/(num_timesteps-1):.2f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
    
    plt.tight_layout()
    plt.show()

# %% 
# Set up parameters
grid_size = 32
num_timesteps = 50
num_simulations = 10
branch_input_dim = grid_size * grid_size
trunk_input_dim = 3  # x, y, t
branch_hidden_layers = [64, 32]
trunk_hidden_layers = [32, 16]
output_dim = 1
num_epochs = 5000
batch_size = 1
learning_rate = 0.001

# %%

# Generate dataset
dataset = generate_dataset(num_simulations, grid_size, num_timesteps)

# %% 
# Create and train the model
model = DeepONetMultipleOutputs(branch_input_dim, trunk_input_dim, branch_hidden_layers, trunk_hidden_layers, output_dim)
train_deeponet(model, dataset, num_epochs, batch_size, learning_rate)

# Visualize results
test_initial_condition = dataset[0][0]
visualize_results(model, test_initial_condition, grid_size, num_timesteps)
# %%

# Add error calculation and visualization
def calculate_and_visualize_error(model, initial_condition, target_solution, grid_size, num_timesteps):
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    branch_input = torch.FloatTensor(initial_condition).unsqueeze(0)
    
    mse_errors = []
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={'projection': '3d'})
    
    for i, t in enumerate([0, num_timesteps//2, num_timesteps-1]):
        trunk_input = torch.FloatTensor(np.column_stack((X.flatten(), Y.flatten(), np.full(X.size, t/num_timesteps))))
        
        output = model(branch_input, trunk_input).detach().numpy().reshape(grid_size, grid_size)
        target = target_solution[t]
        
        # error = np.abs(output - target)
        error = target
        mse = np.mean(error**2)
        mse_errors.append(mse)
        
        surf = axes[i].plot_surface(X, Y, error, cmap='viridis')
        axes[i].set_title(f"Error at t = {t/(num_timesteps-1):.2f}\nMSE = {mse:.6f}")
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_zlabel('Error')
        fig.colorbar(surf, ax=axes[i], shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average MSE across visualized timesteps: {np.mean(mse_errors):.6f}")

# Visualize error
test_initial_condition, test_target_solution = dataset[0]
calculate_and_visualize_error(model, test_initial_condition, test_target_solution, grid_size, num_timesteps)
# %%
