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
    """
    Linear layer with sine non-linearity
    omega_0 is the factor explained in the SIREN paper
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_norm=False,
        is_first=False,
        omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

        self.weight_norm = False
        if weight_norm:
            self.add_weight_norm()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def add_weight_norm(self):
        if not self.weight_norm:
            self.linear = torch.nn.utils.weight_norm(self.linear, name='weight')
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            self.linear = torch.nn.utils.remove_weight_norm(self.linear, name='weight')
            self.weight_norm = False

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=None,
        outermost_linear=False,
        weight_norm=False,
        first_omega_0=30,
        hidden_omega_0=30
    ):
        super().__init__()
        
        # If hidden_features is not a list, make it a list whose elements are all
        # equal to hidden_features and of length hidden_layers, so that it can be
        # handled in the same way
        if not isinstance(hidden_features, list):
            if hidden_layers is not None:
                hidden_features = [hidden_features] * hidden_layers
            else:
                raise ValueError("If hidden_features is not a list, hidden_layers should be specified.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        
        self.net = []
        self.features = [in_features] + hidden_features + [out_features]

        self.net.append(
            SineLayer(
                self.features[0], self.features[1],
                is_first=True,
                omega_0=first_omega_0
            )
        )
        for f_in, f_out in zip(self.features[1:-2], self.features[2:-1]):
            self.net.append(
                SineLayer(
                    f_in, f_out,
                    is_first=False,
                    omega_0=hidden_omega_0
                )
            )
        
        if outermost_linear:
            final_linear = nn.Linear(self.features[-2], self.features[-1])
            
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / self.features[-2]) / hidden_omega_0, 
                    np.sqrt(6 / self.features[-2]) / hidden_omega_0
                )
                
            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    self.features[-2], self.features[-1],
                    is_first=False,
                    omega_0=hidden_omega_0
                )
            )
          
        self.net = nn.Sequential(*self.net)

        self.weight_norm = False
        if weight_norm:
            self.add_weight_norm()

    def add_weight_norm(self):
        if not self.weight_norm:
            for i, mod in enumerate(self.net):
                if isinstance(mod, SineLayer):
                    mod.add_weight_norm()
                else:
                    self.net[i] = torch.nn.utils.weight_norm(mod, name='weight')
            
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            for i, mod in enumerate(self.net):
                if isinstance(mod, SineLayer):
                    mod.remove_weight_norm()
                else:
                    self.net[i] = torch.nn.utils.remove_weight_norm(mod, name='weight')
            
            self.weight_norm = False

    def init_parameters(self):
        weight_norm = self.weight_norm
        if weight_norm:
            self.remove_weight_norm()
        
        self.net[0].linear.reset_parameters()
        self.net[0].init_weights()
        for mod in self.net[1:]:
            if isinstance(mod, SineLayer):
                mod.linear.reset_parameters()
                mod.init_weights()
            else:
                mod.reset_parameters()
                with torch.no_grad():
                    mod.weight.uniform_(
                        -np.sqrt(6 / self.features[-2]) / self.hidden_omega_0, 
                        np.sqrt(6 / self.features[-2]) / self.hidden_omega_0
                    )

        if weight_norm:
            self.add_weight_norm()

    def freeze_parameters(self):
        for param in self.parameters():
            param.required_grad = False

    def ufreeze_parameters(self):
        for param in self.parameters():
            param.required_grad = True

    def forward(self, model_input):
        return torch.sigmoid(self.net(model_input))

    def forward_with_activations(self, model_input, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = model_input['coords']
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return {'model_in': x, 'model_out': activations.popitem(), 'activations': activations}
    
    def __deepcopy__(self, memo=None):
        weight_norm = self.weight_norm
        device = list(self.parameters())[0].device
        self.remove_weight_norm()
        
        c = self.__class__(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            outermost_linear=self.outermost_linear,
            weight_norm=self.weight_norm,
            first_omega_0=self.first_omega_0,
            hidden_omega_0=self.hidden_omega_0
        )
        c.load_state_dict(deepcopy(self.state_dict(), memo))
        c.to(device)

        if weight_norm:
            self.add_weight_norm()
            c.add_weight_norm()
        
        return c

    def get_num_bytes(self):
        size_of_float = 4

        size = 0
        for f_in, f_out in zip(self.features[:-1], self.features[1:]):
            size += f_in * f_out + f_out
        
        size *= size_of_float

        return size

    def save(self, path):
        torch.save(
            {
                'in_features':      self.in_features,
                'out_features':     self.out_features,
                'hidden_features':  self.hidden_features,
                'outermost_linear': self.outermost_linear,
                'first_omega_0':    self.first_omega_0,
                'hidden_omega_0':   self.hidden_omega_0,
                'weight_norm':      self.weight_norm,
                'state_dict':       self.state_dict()
            }, path
        )

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location='cuda:0')
        model = Siren(
            in_features=checkpoint['in_features'],
            hidden_features=checkpoint['hidden_features'],
            out_features=checkpoint['out_features'],
            outermost_linear=checkpoint['outermost_linear'],
            first_omega_0=checkpoint['first_omega_0'],
            hidden_omega_0=checkpoint['hidden_omega_0'],
            weight_norm=checkpoint['weight_norm']
        )
        model.load_state_dict(checkpoint['state_dict'])
        
        return model



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
img_name = 'donuts'
img = np.array(cv2.cvtColor(cv2.imread(f"{img_name}.jpg"), cv2.COLOR_BGR2RGB))
img = img/255
epsilon = 1e-8  # Small constant to avoid log(0)
img_log = np.log(img + epsilon)  # Apply log transformation
print(f"Img: {img_name}.png, Image shape: {img.shape}")
<<<<<<< Updated upstream:decompose_aarush.py
total_steps = 25000
=======
total_steps = 50000
>>>>>>> Stashed changes:decompose.py
steps_til_summary = 200
interpolator_fn = build_2d_sampler(img)
# interpolator_fn = build_2d_sampler(img_log)
batch_size = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

<<<<<<< Updated upstream:decompose_aarush.py
model = Siren(in_features=2, out_features=4, hidden_features=256, 
                hidden_layers=3, outermost_linear=True).to(device)
=======
model = Siren(in_features=2, out_features=4, hidden_features=128, 
                hidden_layers=4, outermost_linear=True, weight_norm=True).to(device)
>>>>>>> Stashed changes:decompose.py
best_loss_combined = float("inf")


optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
<<<<<<< Updated upstream:decompose_aarush.py
=======

>>>>>>> Stashed changes:decompose.py
writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
print("Training All Integral Field")
et = time.time()    
for step in range(total_steps):
    model_input, ground_truth = generate_training_samples_2d(batch_size, interpolator_fn, img)
    out = model(model_input)
    output = out[:, 3:] * out[:, :3]
    # output = out[:, 3:] + out[:, :3]
    if loss_type=="l1":
        loss_f = torch.nn.functional.smooth_l1_loss(ground_truth, output)
    elif loss_type=="l2":
        loss_f = (((ground_truth - output))**2).mean()

    grads = (vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1))(model_input[:, :1], model_input[:, 1:])).reshape(-1, 4)
    reflectance_grad = grads[:, :3]
    shading_grad = grads[:, 3:]

<<<<<<< Updated upstream:decompose_aarush.py
    loss = 2*loss_f + 1e-9 * reflectance_grad.abs().sum()
=======
    loss = 8*loss_f + 1e-9 * reflectance_grad.abs().sum() 
    # loss = loss_f 
>>>>>>> Stashed changes:decompose.py

    optim.zero_grad()
    loss.backward()
    optim.step()

    writer.add_scalar('f loss', loss_f.item(), step)
    if not step % steps_til_summary:
        print("Step", step, '| combined loss:', loss.item(), '| f loss:', loss_f.item())

    # Ensure 'weights' directory exists
    os.makedirs("weights", exist_ok=True)

    if loss.item() < best_loss_combined:
        torch.save(model.state_dict(), f'weights/siren_{img_name}_{loss_type}.pth')
        best_loss_combined = loss.item()
        # Save weights and log the best loss
    os.makedirs("weights", exist_ok=True)  # Ensure 'weights' directory exists
    if loss.item() < best_loss_combined:
        torch.save(model.state_dict(), f'weights/siren_{img_name}_{loss_type}.pth')
        best_loss_combined = loss.item()

<<<<<<< Updated upstream:decompose_aarush.py
# Final Reconstruction and Visualization
=======
>>>>>>> Stashed changes:decompose.py
shape = img.shape
xy_grid = get_grid(shape[0], shape[1]).to(device)
generated = model(xy_grid)
# generated = torch.exp(generated) - epsilon 

print("albedo: min max", generated[:, :3].min(), generated[:, :3].max())
print("shading: min max", generated[:, 3:].min(), generated[:, 3:].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

<<<<<<< Updated upstream:decompose_aarush.py
axes[0].imshow((generated[:, 3:]*generated[:, :3]).reshape(*shape).cpu().detach().numpy())
axes[0].set_title("Reconstruction")

z = (generated[:, :3] - generated[:, :3].min())/(generated[:, :3].max() - generated[:, :3].min())
axes[1].imshow((generated[:, :3]).reshape(*shape).cpu().detach().numpy())
axes[1].set_title("Albedo")
axes[2].imshow((generated[:, 3:]).reshape(shape[0], shape[1], 1).cpu().detach().numpy(), cmap="gray")
axes[2].set_title("Shading")
plt.savefig("out.png")
=======
z = generated[:, 3:]*generated[:, :3]
# z = (z - z.min())/(z.max() - z.min())
axes[0].imshow((z).reshape(*shape).cpu().detach().numpy())
axes[0].set_title("reconstruction")

z = generated[:, :3]
# z = (z - z.min())/(z.max() - z.min())
axes[1].imshow((z).reshape(*shape).cpu().detach().numpy())
axes[1].set_title("albedo")

z = generated[:, 3:]
# z = (z - z.min())/(z.max() - z.min())
axes[2].imshow((z).reshape(shape[0], shape[1], 1).cpu().detach().numpy(), cmap="gray")
axes[2].set_title("shading")
plt.savefig("outr.png")
>>>>>>> Stashed changes:decompose.py
plt.tight_layout()
plt.show()
