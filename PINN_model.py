

############## u(x)=C1cos(ωx)+C2sin(ωx)+C3cos(2ωx) + C4sin(2ωx) +C5  


import torch
import torch.nn as nn
import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_torch(10000)
torch.cuda.empty_cache()


data_path = "#name.mat"
mat_data = sio.loadmat(data_path)
data_combined = mat_data["name"].flatten()
data_combinedf12 = np.flip(data_combined)


# Known data regions
x1 = np.arange(1, 50001)  # X values for data1
u1 = data_combinedf12[0:50000]  # Data1
x2 = np.arange(200001, 250001)  # X values for data2
u2 = data_combinedf12[200000:250000]  # Data2

# Scale x-values for consistent training
scaler_x = StandardScaler()
x_combined = np.arange(1, 250001)  # Full domain x-values
x_scaled = scaler_x.fit_transform(x_combined.reshape(-1, 1)).flatten()  

x1_scaled = x_scaled[:50000]
x2_scaled = x_scaled[200000:250000]
u_min = min(u1.min(), u2.min())
u_max = max(u1.max(), u2.max())


u1 = (u1 - u_min) / (u_max - u_min)
u2 = (u2 - u_min) / (u_max - u_min)


known_x = torch.tensor(
    np.concatenate([x1_scaled, x2_scaled]).reshape(-1, 1), dtype=torch.float32
).to(DEVICE)
known_u = torch.tensor(
    np.concatenate([u1, u2]).reshape(-1, 1), dtype=torch.float32
).to(DEVICE)

# Boundary data tensors
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
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )
        # Learnable parameters
        self.C1 = nn.Parameter(torch.randn(1))
        self.C2 = nn.Parameter(torch.randn(1))
        self.C3 = nn.Parameter(torch.randn(1))
        self.C4 = nn.Parameter(torch.randn(1))
        self.rawomega = nn.Parameter(torch.randn(1))

    def forward(self, x):
        omega = torch.sigmoid(self.rawomega) * 0.001  # Constrain omega in [0, 0.01]
        net_output = self.model(x)
        constrained_output = (
                self.C1 * torch.cos(omega * x)
                + self.C2 * torch.sin(omega * x)
                + self.C3
                + self.C4
        )
        return net_output + constrained_output

    def get_params(self):
        omega = torch.sigmoid(self.rawomega) * 0.001  # Constrain omega in [0, 0.01]
        return self.C1, self.C2, self.C3, self.C4, omega

# Compute derivatives
def derivative(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True
    )[0]


def PDE_loss_fourth_order_with_C4_and_omega(model, num_pde_points=10000):
    C1, C2, C3, C4, omega = model.get_params()
    x = torch.rand((num_pde_points, 1), requires_grad=True).to(DEVICE)
    u = model(x)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)
    u_xxx = derivative(u_xx, x)
    u_xxxx = derivative(u_xxx, x)
    pde_residual = u_xxxx - omega**4 * (u - C3 - C4)
    return torch.mean(pde_residual**2)



def combined_loss_with_C3_and_omega(model, num_pde_points=10000):
    pde_loss = PDE_loss_fourth_order_with_C4_and_omega(model, num_pde_points)
    predicted_u = model(known_x)
    data_loss = mse_loss(predicted_u, known_u)
    predicted_boundary_u = model(boundary_x)
    boundary_loss = mse_loss(predicted_boundary_u, boundary_u)
    return pde_loss + data_loss + boundary_loss


model = PINN().to(DEVICE)


adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lbfgs_optimizer = torch.optim.LBFGS(
    model.parameters(), max_iter=5000, tolerance_grad=1e-7, tolerance_change=1e-8
)
mse_loss = nn.MSELoss()


ADAM_EPOCHS = 10000
for epoch in range(ADAM_EPOCHS):
    adam_optimizer.zero_grad()
    loss = combined_loss_with_C3_and_omega(model)
    loss.backward()
    adam_optimizer.step()

    if (epoch + 1) % 1000 == 0:
        C1, C2, C3, C4, omega = model.get_params()
        print(
            f"[Adam Training] Epoch {epoch + 1}/{ADAM_EPOCHS}, Loss: {loss.item()}, "
            f"C1: {C1.item()}, C2: {C2.item()}, C3: {C3.item()}, C4: {C4.item()}, omega: {omega.item()}"
        )

# Training with L-BFGS
print("Switching to L-BFGS optimizer for fine-tuning...")


def closure():
    lbfgs_optimizer.zero_grad()
    loss = combined_loss_with_C3_and_omega(model)
    loss.backward()
    return loss


lbfgs_optimizer.step(closure)


C1, C2, C3, C4, omega = model.get_params()
print(
    f"Final Parameters: C1: {C1.item()}, C2: {C2.item()}, C3: {C3.item()}, "
    f"C4: {C4.item()}, omega: {omega.item()}"
)

with torch.no_grad():
    x_test = torch.tensor(x_scaled, dtype=torch.float32).view(-1, 1).to(DEVICE)
    u_pred = model(x_test).cpu().numpy()

u_pred = u_pred * (u_max - u_min) + u_min


sio.savemat("data_combined_imf1_1124_noflip_ANaN_imf1.mat", {"u_pred": u_pred})


plt.figure(figsize=(12, 6))
plt.plot(data_combined, label="Original Data (Ground Truth)", color="blue")
plt.plot(u_pred, label="PINN Prediction", color="red", linestyle="--")
plt.xlabel("X (Original Scale)")
plt.ylabel("Value")
plt.title("PINN Prediction vs Original Data")
plt.legend()
plt.grid()
plt.show()


######  ==================  比较真实值 ==================
u_pred = u_pred.flatten()
data_path1 = "rawdata_A.mat"
mat_data1 = sio.loadmat(data_path1)
raw_data = mat_data1["A"].flatten()
raw_data = raw_data
print('raw_data.shape', raw_data.shape)
differ = raw_data - u_pred
# # sio.savemat('u_pred_imf1_differ.mat', {'differ': differ})
plt.figure(figsize=(12, 6))
plt.plot(u_pred, label="PINN Prediction", color='blue')
plt.plot(raw_data, label="Original Data", color='red')
plt.xlabel("X (Original Scale)")
plt.ylabel("value")
plt.title("error curve")
plt.legend()
plt.grid()
plt.show()



