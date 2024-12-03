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
from networks import CoordinateNet_ordinary

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
print(f"Img: {img_name}.jpg, Image shape: {img.shape}")
total_steps = 100000
steps_til_summary = 200
# interpolator_fn = build_2d_sampler(img)
interpolator_fn = build_2d_sampler(img)
batch_size = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CoordinateNet_ordinary(4,
                          "swish",
                          2,
                          128,
                          3,
                          8).to(device)
best_loss_combined = float("inf")


optim = torch.optim.Adam(lr=1e-3, params=model.parameters())

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

    # grads = (vmap(jacrev(jacfwd(lambda a, b: model(torch.cat([a, b], -1)), argnums=0), argnums=1))(model_input[:, :1], model_input[:, 1:])).reshape(-1, 4)
    # reflectance_grad = grads[:, :3]
    # shading_grad = grads[:, 3:]

    # loss = loss_f + 1e-10 * shading_grad.norm(p=2, dim=-1).mean()
    loss = loss_f 

    optim.zero_grad()
    loss.backward()
    optim.step()


    writer.add_scalar('f loss',
                        loss_f.item(),
                        step)
    if not step % steps_til_summary:
        print("Step", step, '| combined loss:', loss.item(), '| f loss:', loss_f.item())

    # if loss.item() < best_loss_combined:
    #     torch.save(model.state_dict(), f'weights/siren_{img_name}_{loss_type}.pth')
    #     best_loss_combined = loss.item()

shape = img.shape
xy_grid = get_grid(shape[0], shape[1]).to(device)
generated = model(xy_grid)
# generated = torch.exp(generated) - epsilon 

print("albedo: min max", generated[:, :3].min(), generated[:, :3].max())
print("shading: min max", generated[:, 3:].min(), generated[:, 3:].max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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
plt.savefig("outdpe.png")
plt.tight_layout()
plt.show()
