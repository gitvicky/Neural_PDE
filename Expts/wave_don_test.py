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

def generate_wave_data(num_simulations, nx, ny, nt, dx, dy, dt, c):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    t = np.linspace(0, 1, nt)
    
    X, Y, T = np.meshgrid(x, y, t, indexing='ij')
    
    data = []
    for _ in range(num_simulations):
        # Random initial conditions
        u0 = np.random.randn(nx, ny) * 0.1
        u1 = np.random.randn(nx, ny) * 0.1
        
        u = np.zeros((nx, ny, nt))
        u[:, :, 0] = u0
        u[:, :, 1] = u1
        
        # Finite difference method to solve the wave equation
        for n in range(2, nt):
            u[1:-1, 1:-1, n] = 2*u[1:-1, 1:-1, n-1] - u[1:-1, 1:-1, n-2] + \
                               (c*dt/dx)**2 * (u[2:, 1:-1, n-1] - 2*u[1:-1, 1:-1, n-1] + u[:-2, 1:-1, n-1]) + \
                               (c*dt/dy)**2 * (u[1:-1, 2:, n-1] - 2*u[1:-1, 1:-1, n-1] + u[1:-1, :-2, n-1])
        
        data.append(u)
    
    return X, Y, T, np.array(data)

def prepare_data(X, Y, T, data, train_ratio=0.8):
    nx, ny, nt = X.shape
    num_simulations = data.shape[0]
    
    branch_input = data[:, :, :, 0].reshape(num_simulations, -1)
    trunk_input = np.column_stack((X.ravel(), Y.ravel(), T.ravel()))
    output = data.reshape(num_simulations, -1)
    
    # Split data into train and test sets
    train_size = int(train_ratio * num_simulations)
    train_branch = torch.FloatTensor(branch_input[:train_size])
    train_trunk = torch.FloatTensor(trunk_input)
    train_output = torch.FloatTensor(output[:train_size])
    
    test_branch = torch.FloatTensor(branch_input[train_size:])
    test_trunk = torch.FloatTensor(trunk_input)
    test_output = torch.FloatTensor(output[train_size:])
    
    return (train_branch, train_trunk, train_output), (test_branch, test_trunk, test_output)

def train_model(model, train_data, test_data, num_epochs, batch_size, learning_rate):
    train_branch, train_trunk, train_output = train_data
    test_branch, test_trunk, test_output = test_data
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, train_branch.size(0), batch_size):
            batch_branch = train_branch[i:i+batch_size]
            batch_output = train_output[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(batch_branch, train_trunk)
            loss = criterion(pred, batch_output)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            test_pred = model(test_branch, test_trunk)
            test_loss = criterion(test_pred, test_output)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Test Loss: {test_loss:.4f}")

def plot_results(X, Y, T, true_data, pred_data, time_step):
    fig = plt.figure(figsize=(20, 5))
    
    # True solution
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X[:,:,time_step], Y[:,:,time_step], true_data[:,:,time_step], cmap='viridis')
    ax1.set_title('True Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    
    # Predicted solution
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X[:,:,time_step], Y[:,:,time_step], pred_data[:,:,time_step], cmap='viridis')
    ax2.set_title('Predicted Solution')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    
    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    error = np.abs(true_data[:,:,time_step] - pred_data[:,:,time_step])
    ax3.plot_surface(X[:,:,time_step], Y[:,:,time_step], error, cmap='viridis')
    ax3.set_title('Absolute Error')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Error')
    
    plt.tight_layout()
    plt.show()

# %% 
# Main execution
if __name__ == "__main__":
    # Parameters
    num_simulations = 10
    nx, ny, nt = 50, 50, 100
    dx = dy = 1.0 / (nx - 1)
    dt = 1.0 / (nt - 1)
    c = 1.0  # Wave speed
    
    # Generate data
    X, Y, T, data = generate_wave_data(num_simulations, nx, ny, nt, dx, dy, dt, c)
    
    # Prepare data
    train_data, test_data = prepare_data(X, Y, T, data)
    
    # Model parameters
    branch_input_dim = nx * ny
    trunk_input_dim = 3  # x, y, t
    branch_hidden_layers = [64, 32]
    trunk_hidden_layers = [32, 16]
    output_dim = nx * ny * nt
    
    # Create and train the model
    model = DeepONetMultipleOutputs(branch_input_dim, trunk_input_dim, branch_hidden_layers, trunk_hidden_layers, output_dim)
    train_model(model, train_data, test_data, num_epochs=100, batch_size=16, learning_rate=0.001)
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_branch, test_trunk, test_output = test_data
        pred_output = model(test_branch, test_trunk)
    
    # Reshape the data for plotting
    true_data = test_output[0].numpy().reshape(nx, ny, nt)
    pred_data = pred_output[0].numpy().reshape(nx, ny, nt)
    
    # Plot results at a specific time step
    plot_results(X, Y, T, true_data, pred_data, time_step=50)
# %%
