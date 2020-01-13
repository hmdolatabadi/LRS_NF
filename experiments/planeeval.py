import argparse
import json
import numpy as np
import torch
import os

from matplotlib import cm, pyplot as plt
import matplotlib.ticker as ticker
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils import data
from tqdm import tqdm

import data as data_
import nn as nn_
import utils

from experiments import cutils
from nde import distributions, flows, transforms

dataset_name = 'rings'
path = os.path.join(cutils.get_final_root(), '{}-final.json'.format(dataset_name))
with open(path) as file:
    dictionary = json.load(file)
args = argparse.Namespace(**dictionary)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.use_gpu:
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')

# create data
train_dataset = data_.load_plane_dataset(args.dataset_name, args.n_data_points)
train_loader = data_.InfiniteLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_epochs=None
)

# Generate test grid data
num_points_per_axis = 512
limit = 4
bounds = np.array([
    [-limit, limit],
    [-limit, limit]
])
grid_dataset = data_.TestGridDataset(
    num_points_per_axis=num_points_per_axis,
    bounds=bounds
)
grid_loader = data.DataLoader(
    dataset=grid_dataset,
    batch_size=1000,
    drop_last=False
)
dim = 2

# create model
# distribution = distributions.TweakedUniform(
#     low=torch.zeros(dim),
#     high=torch.ones(dim)
# )
distribution = distributions.StandardNormal((2,))
# transform = transforms.CompositeTransform([
#     transforms.Sigmoid(),
#     transforms.PiecewiseRationalQuadraticCouplingTransform(
#             mask=utils.create_alternating_binary_mask(features=dim, even=True),
#             transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
#                 in_features=in_features,
#                 out_features=out_features,
#                 hidden_features=32,
#                 num_blocks=2,
#                 use_batch_norm=True
#             ),
#             num_bins=args.num_bins,
#             apply_unconditional_transform=False
#     ),
#     transforms.PiecewiseRationalQuadraticCouplingTransform(
#             mask=utils.create_alternating_binary_mask(features=dim, even=False),
#             transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
#                 in_features=in_features,
#                 out_features=out_features,
#                 hidden_features=32,
#                 num_blocks=2,
#                 use_batch_norm=True
#             ),
#             num_bins=args.num_bins,
#             apply_unconditional_transform=False
#     )
# ])

transform = transforms.CompositeTransform([
    # transforms.Sigmoid(),
    transforms.PiecewiseRationalLinearCouplingTransform(
            mask=utils.create_alternating_binary_mask(features=dim, even=True),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=32,
                num_blocks=2,
                use_batch_norm=True
            ),
            tails='linear',
            tail_bound=5,
            num_bins=args.num_bins,
            apply_unconditional_transform=False
    ),
    transforms.PiecewiseRationalLinearCouplingTransform(
            mask=utils.create_alternating_binary_mask(features=dim, even=False),
            transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=32,
                num_blocks=2,
                use_batch_norm=True
            ),
            tails='linear',
            tail_bound=5,
            num_bins=args.num_bins,
            apply_unconditional_transform=False
    )
])

flow = flows.Flow(transform, distribution).to(device)
path = os.path.join(cutils.get_final_root(), '{}-final.t'.format(dataset_name))
state_dict = torch.load(path)
flow.load_state_dict(state_dict)
flow.eval()

n_params = utils.get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

log_density_np = []
for batch in grid_loader:
    batch = batch.to(device)
    _, log_density = flow.log_prob(batch)
    log_density_np = np.concatenate(
        (log_density_np, utils.tensor2numpy(log_density))
    )

vmax = np.exp(log_density_np).max() * 0.7
cmap = cm.magma
# plot data
figure, axes = plt.subplots(1, 1, figsize=(3.125, 3.125))
axes.hist2d(utils.tensor2numpy(train_dataset.data[:, 0]),
            utils.tensor2numpy(train_dataset.data[:, 1]),
            range=bounds, bins=512, rasterized=False, normed=True, cmap='inferno')
axes.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
axes.xaxis.set_major_locator(ticker.NullLocator())
axes.yaxis.set_major_locator(ticker.NullLocator())
path = os.path.join(cutils.get_output_root(), '{}-data.png'.format(dataset_name))
plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=320)
plt.close()

# plot density
figure, axes = plt.subplots(1, 1, figsize=(3.125, 3.125))
# axes.pcolormesh(grid_dataset.X, grid_dataset.Y,
#                 np.exp(log_density_np).reshape(grid_dataset.X.shape),
#                 cmap=cmap, vmin=0)
axes.imshow(np.exp(log_density_np).reshape(grid_dataset.X.shape), cmap='inferno')
axes.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
axes.xaxis.set_major_locator(ticker.NullLocator())
axes.yaxis.set_major_locator(ticker.NullLocator())
path = os.path.join(cutils.get_output_root(), '{}-density.png'.format(dataset_name))
plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=320)
plt.close()

# plot samples
figure, axes = plt.subplots(1, 1, figsize=(3.125, 3.125))
with torch.no_grad():
    samples = utils.tensor2numpy(
        flow.sample(num_samples=int(1e6), batch_size=int(1e5)))
axes.hist2d(samples[:, 0], samples[:, 1],
               range=bounds, bins=512, cmap='inferno', rasterized=False, normed=True)
axes.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)
axes.xaxis.set_major_locator(ticker.NullLocator())
axes.yaxis.set_major_locator(ticker.NullLocator())
path = os.path.join(cutils.get_output_root(), '{}-samples.png'.format(dataset_name))
plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=320)
plt.close()
