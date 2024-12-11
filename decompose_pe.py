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
from networks import CoordinateNet_ordinary, Siren
from torch.autograd import grad

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
    random_samples_np = np.random.uniform(low=-1, high=1, size=[batch_size, 2])
    sample_coord_x = map_range(random_samples_np[:, 0], (-1, 1), (0, H - 1)).reshape(-1,1)
    sample_coord_y = map_range(random_samples_np[:, 1], (-1, 1), (0, W - 1)).reshape(-1,1)
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
    tensors_x = torch.linspace(-1, 1, steps=sidelenx)
    tensors_y = torch.linspace(-1, 1, steps=sideleny)
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
img_path = r'dataset\IIW'
img_name = r'lady'
img = np.array(cv2.cvtColor(cv2.imread(f"{os.path.join(img_path, img_name)}.jpg"), cv2.COLOR_BGR2RGB))
img = img/255
epsilon = 1e-8  # Small constant to avoid log(0)
img_log = np.log(img + epsilon)  # Apply log transformation
print(f"Img: {img_name}.jpg, Image shape: {img.shape}")
total_steps = 20000
steps_til_summary = 200
# interpolator_fn = build_2d_sampler(img)
interpolator_fn = build_2d_sampler(img)
batch_size = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = CoordinateNet_ordinary(4,
#                           "swish",
#                           2,
#                           128,
#                           3,
#                           8).to(device)

model = Siren(in_features=2, out_features=4, hidden_features=128, 
                hidden_layers=4, outermost_linear=True, weight_norm=True).to(device)
best_loss_combined = float("inf")


optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

writer = SummaryWriter(f'runs/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
print("Training All Integral Field")
et = time.time()    
for step in range(total_steps):
    model_input, ground_truth = generate_training_samples_2d(batch_size, interpolator_fn, img)
    model_input.requires_grad_(True)

    out = model(model_input)
    albedo, shading = torch.split(out, 3, dim=-1)
    reconstruction = albedo * shading
    if loss_type=="l1":
        loss_f = torch.nn.functional.smooth_l1_loss(ground_truth, reconstruction)
    elif loss_type=="l2":
        loss_f = F.mse_loss(reconstruction, ground_truth)

    image_chromaticity = ground_truth / (ground_truth.norm(dim=-1, keepdim=True) + 1e-8)
    albedo_chromaticity = albedo / (albedo.norm(dim=-1, keepdim=True) + 1e-8)
    chromaticity_loss = ((image_chromaticity - albedo_chromaticity) ** 2).mean()
    shading_non_negativity_loss = torch.relu(-shading).pow(2).mean()

    reflectance_grad = torch.autograd.grad(albedo.abs().sum(), model_input,  create_graph=True)[0]
    reflectance_grad_mag = reflectance_grad.norm(p=2, dim=-1)
    loss = loss_f + 0.1 * chromaticity_loss + 0.1*shading_non_negativity_loss +  5e-6*(reflectance_grad_mag).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


    writer.add_scalar('f loss',
                        loss_f.item(),
                        step)
    if not step % steps_til_summary:
        print("Step", step, '| combined loss:', loss.item(), '| f loss:', loss_f.item())

shape = img.shape
xy_grid = get_grid(shape[0], shape[1]).to(device)
generated = model(xy_grid)

print("albedo: min max", generated[:, :3].min(), generated[:, :3].max())
print("shading: min max", generated[:, 3:].min(), generated[:, 3:].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

z = generated[:, 3:]*generated[:, :3]
axes[0].imshow((z).reshape(*shape).cpu().detach().numpy())
axes[0].set_title("reconstruction")
cv2.imwrite(fr"results\IIW\Priors\{img_name}.png", 255*cv2.cvtColor((z).reshape(*shape).cpu().detach().numpy(), cv2.COLOR_BGR2RGB))

z = generated[:, :3]
axes[1].imshow((z).reshape(*shape).cpu().detach().numpy())
axes[1].set_title("albedo")
cv2.imwrite(fr"results\IIW\Priors\{img_name}_reflectance.png", 255*cv2.cvtColor((z).reshape(*shape).cpu().detach().numpy(), cv2.COLOR_BGR2RGB))

z = generated[:, 3:]
axes[2].imshow((z).reshape(shape[0], shape[1], 1).cpu().detach().numpy(), cmap="gray")
axes[2].set_title("shading")
cv2.imwrite(fr"results\IIW\Priors\{img_name}_shading.png", 255*(z).reshape(shape[0], shape[1], 1).cpu().detach().numpy())
plt.savefig(fr"results/IIW/Priors/{img_name}_combined.png")
plt.tight_layout()
plt.show()
