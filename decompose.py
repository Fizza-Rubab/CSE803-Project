import torch
from torch.func import vmap, jacfwd, jacrev
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
from torch.func import vmap, jacfwd, jacrev
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Process
from datetime import datetime
import cv2


class SineLayer(nn.Module):    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6,
                 include_input=True,
                 log_sampling=True,
                 normalize=False,
                 input_dim=3,
                 gaussian_pe=False,
                 norm_exp=1,
                 gaussian_variance=38):

        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(0.0, self.num_encoding_functions - 1,
                                                             self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (self.num_encoding_functions - 1),
                                                      self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1 / self.frequency_bands ** norm_exp)

    def forward(self, tensor) -> torch.Tensor:
        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:

                    if self.normalization is not None:
                        encoding.append(self.normalization[idx] * func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = torch.sigmoid(self.net(coords))
        return output   


def map_range(values, old_range, new_range):
    new_width = (new_range[1] - new_range[0])
    old_width = (old_range[1] - old_range[0])
    return (((values - old_range[0]) * new_width) / old_width) + new_range[0]
    
def build_2d_sampler(data, method='linear'):
    x = np.linspace(0, data.shape[0] - 1, data.shape[0])
    y = np.linspace(0, data.shape[1] - 1, data.shape[1])
    return RegularGridInterpolator((x, y), data, method=method)


def generate_training_samples_2d(batch_size, interpolator_fn, img, precision=32):
    H, W = img.shape[:2]
    random_samples_np = np.random.uniform(low=0, high=1, size=[batch_size, 2])
    sample_coord_x = map_range(random_samples_np[:, 0], (0, 1), (0, H - 1)).reshape(-1,1)
    sample_coord_y = map_range(random_samples_np[:, 1], (0, 1), (0, W - 1)).reshape(-1,1)
    sample_coord =  np.concatenate([sample_coord_x, sample_coord_y], axis=1)
    input_tensor = torch.unsqueeze(torch.from_numpy(random_samples_np), 0).cuda()
    bi_sampled = interpolator_fn(sample_coord)
    bi_sampled = torch.from_numpy(bi_sampled).cuda()
    rgb_data = bi_sampled.contiguous().view(-1, 3)
    input_tensor = input_tensor.view(-1, 2)
    input_tensor = input_tensor.float() if precision == 32 else input_tensor.double()
    rgb_data = rgb_data.float() if precision == 32 else rgb_data.double()
    return input_tensor.cuda(), rgb_data.cuda()

def get_grid(sidelenx,  sideleny, dim=2):
    tensors_x = torch.linspace(0, 1, steps=sidelenx)
    tensors_y = torch.linspace(0, 1, steps=sideleny)
    tensors = (tensors_x, tensors_y,)
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def tensor_to_numpy(tensor: torch.Tensor, shape) -> np.ndarray:
    tensor = tensor * 256
    tensor[tensor > 255] = 255
    tensor[tensor < 0] = 0
    tensor = tensor.type(torch.uint8).reshape(*shape).cpu().numpy()
    return tensor

loss_type = "l2"
img_name = '90785'
img = np.array(cv2.cvtColor(cv2.imread(f"{img_name}.png"), cv2.COLOR_BGR2RGB))
img = img/255
print(f"Img: {img_name}.png, Image shape: {img.shape}")
total_steps = 11000
steps_til_summary = 200
interpolator_fn = build_2d_sampler(img)
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Siren(in_features=2, out_features=4, hidden_features=128, 
                hidden_layers=3, outermost_linear=True).to(device)
best_loss_combined = float("inf")


optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5000, gamma=0.8)

writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
print("Training All Integral Field")
et = time.time()    
for step in range(total_steps):
    model_input, ground_truth = generate_training_samples_2d(batch_size, interpolator_fn, img)
    out = model(model_input)
    # print("model input output", model_input.shape, out.shape)
    output = out[:, 3:] * out[:, :3]
    # print("output", output.shape)
    if loss_type=="l1":
        loss_f = torch.nn.functional.smooth_l1_loss(ground_truth, output)
    elif loss_type=="l2":
        loss_f = (((ground_truth - output))**2).mean()

    grads = (vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1))(model_input[:, :1], model_input[:, 1:])).reshape(-1, 4)
    reflectance_grad = grads[:, :3]
    shading_grad = grads[:, 3:]

    # loss = loss_f + 1e-6 * reflectance_grad.abs().sum() + 1e-7 * shading_grad.norm(p=2, dim=-1).mean()
    loss = loss_f + 1e-9 * reflectance_grad.abs().sum()

    optim.zero_grad()
    loss.backward()
    optim.step()
    # scheduler.step()


    writer.add_scalar('f loss',
                        loss_f.item(),
                        step)
    if not step % steps_til_summary:
        print("Step", step, '| combined loss:', loss.item(), '| f loss:', loss_f.item())

    if loss.item() < best_loss_combined:
        torch.save(model.state_dict(), f'weights/siren_{img_name}_{loss_type}.pth')
        best_loss_combined = loss.item()

    # if step>0 and step % 5000 == 0:
    #     shape = img.shape
    #     xy_grid = get_grid(shape[0], shape[1]).to(device)
    #     generated = model(xy_grid)
        
    #     print("albedo: min max", generated[:, :3].min(), generated[:, :3].max())
    #     print("shading: min max", generated[:, 3:].min(), generated[:, 3:].max())

    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
    #     axes[0].imshow((generated[:, 3:]*generated[:, :3]).reshape(*shape).cpu().detach().numpy())
    #     axes[0].set_title("reconstruction")

    #     z = (generated[:, :3] - generated[:, :3].min())/(generated[:, :3].max() - generated[:, :3].min())
    #     axes[1].imshow((generated[:, :3]).reshape(*shape).cpu().detach().numpy())
    #     axes[1].set_title("albedo")
    #     axes[2].imshow((generated[:, 3:]).reshape(shape[0], shape[1], 1).cpu().detach().numpy(), cmap="gray")
    #     axes[2].set_title("shading")
    #     plt.savefig("out.png")
    #     plt.tight_layout()
    #     plt.show()

shape = img.shape
xy_grid = get_grid(shape[0], shape[1]).to(device)
generated = model(xy_grid)

print("albedo: min max", generated[:, :3].min(), generated[:, :3].max())
print("shading: min max", generated[:, 3:].min(), generated[:, 3:].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow((generated[:, 3:]*generated[:, :3]).reshape(*shape).cpu().detach().numpy())
axes[0].set_title("reconstruction")

z = (generated[:, :3] - generated[:, :3].min())/(generated[:, :3].max() - generated[:, :3].min())
axes[1].imshow((generated[:, :3]).reshape(*shape).cpu().detach().numpy())
axes[1].set_title("albedo")
axes[2].imshow((generated[:, 3:]).reshape(shape[0], shape[1], 1).cpu().detach().numpy(), cmap="gray")
axes[2].set_title("shading")
plt.savefig("out.png")
plt.tight_layout()
plt.show()
