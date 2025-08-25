# Reference: https://github.com/MasterMeep/Heat-Equation-PINN-prediction/tree/main


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
data_path = "#name.mat"
mat_data = sio.loadmat(data_path)
data_combinedf12 = mat_data["name"].flatten()

x1 = np.arange(1, 50001)
u1 = data_combinedf12[0:50000]
x2 = np.arange(200001, 250001)
u2 = data_combinedf12[200000:250000]

scaler_x = StandardScaler()
x_combined = np.arange(1, 250001)
x_scaled = scaler_x.fit_transform(x_combined.reshape(-1, 1)).flatten()

x1_scaled = x_scaled[:50000]
x2_scaled = x_scaled[200000:250000]

u_min = min(u1.min(), u2.min())
u_max = min(u1.max(), u2.max())

u1 = (u1 - u_min) / (u_max - u_min)
u2 = (u2 - u_min) / (u_max - u_min)

batch_size = 500


known_x = torch.tensor(
    np.concatenate([x1_scaled, x2_scaled]).reshape(-1, 1), dtype=torch.float32
).to(DEVICE)
known_u = torch.tensor(
    np.concatenate([u1, u2]).reshape(-1, 1), dtype=torch.float32
).to(DEVICE)

boundary_x = torch.tensor(
    [x1_scaled[0], x1_scaled[-1], x2_scaled[0], x2_scaled[-1]], dtype=torch.float32
).view(-1, 1).to(DEVICE)
boundary_u = torch.tensor(
    [u1[0], u1[-1], u2[0], u2[-1]], dtype=torch.float32
).view(-1, 1).to(DEVICE)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.model(x)

def derivative(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )[0]


def PDE_loss_fifth_order(model, num_pde_points=10000, batch_size=batch_size):
    total_residual = 0.0
    for _ in range(num_pde_points // batch_size):
        x = torch.rand((batch_size, 1), requires_grad=True).to(DEVICE)
        u = model(x)

        # Compute derivatives
        u_x = derivative(u, x)
        u_xx = derivative(u_x, x)
        u_xxx = derivative(u_xx, x)
        u_xxxx = derivative(u_xxx, x)
        u_xxxxx = derivative(u_xxxx, x)
        u_xxxxxx = derivative(u_xxxxx, x)
        # u_xxxxxxx = derivative(u_xxxxxx, x)
        # u_xxxxxxxx = derivative(u_xxxxxxx, x)
        pde_residual = u_xxxxxx

        total_residual += torch.sum(pde_residual ** 2)

    return total_residual / num_pde_points


def combined_loss(model, num_pde_points=10000, batch_size=batch_size):
    pde_loss_value = PDE_loss_fifth_order(model, num_pde_points, batch_size)


    predicted_u = model(known_x)
    data_loss = mse_loss(predicted_u, known_u)


    predicted_boundary_u = model(boundary_x)
    boundary_loss = mse_loss(predicted_boundary_u, boundary_u)

    total_loss = pde_loss_value + 10 * data_loss + 10.0 * boundary_loss
    return total_loss



model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

EPOCHS = 2000

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = combined_loss(model, batch_size=batch_size)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")


with torch.no_grad():
    x_test = torch.tensor(x_scaled, dtype=torch.float32).view(-1, 1).to(DEVICE)
    BATCH_SIZE_PREDICT = 5000
    u_pred = []

    for i in range(0, len(x_test), BATCH_SIZE_PREDICT):
        batch_x = x_test[i:i + BATCH_SIZE_PREDICT]
        u_pred.append(model(batch_x).cpu().numpy())

    u_pred = np.concatenate(u_pred, axis=0)

u_pred = u_pred*(u_max - u_min) + u_min

plt.figure(figsize=(12, 6))
plt.plot(data_combinedf12, label="Original Data (Ground Truth)", color='blue')
plt.plot(u_pred, label="PINN Prediction", color='red', linestyle="--")
plt.xlabel("X (Original Scale)")
plt.ylabel("Value")
plt.title("PINN Prediction vs Original Data")
plt.legend()
plt.grid()
plt.show()


