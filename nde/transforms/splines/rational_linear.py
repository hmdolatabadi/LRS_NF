import torch
from torch.nn import functional as F

import utils
import numpy as np

from nde import transforms

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def unconstrained_rational_linear_spline(inputs,
                                         unnormalized_widths,
                                         unnormalized_heights,
                                         unnormalized_derivatives,
                                         unnormalized_lambdas,
                                         inverse=False,
                                         tails='linear',
                                         tail_bound=1.,
                                         min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                         min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                         min_derivative=DEFAULT_MIN_DERIVATIVE):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs   = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_linear_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        unnormalized_lambdas=unnormalized_lambdas[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet

def rational_linear_spline(inputs,
                           unnormalized_widths,
                           unnormalized_heights,
                           unnormalized_derivatives,
                           unnormalized_lambdas,
                           inverse=False,
                           left=0., right=1., bottom=0., top=1.,
                           min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                           min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                           min_derivative=DEFAULT_MIN_DERIVATIVE):
    assert inputs.numel() != 0
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise transforms.InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left

    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = utils.searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = utils.searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    lambdas = 0.95 * torch.sigmoid(unnormalized_lambdas) + 0.025

    lam = lambdas.gather(-1, bin_idx)[..., 0]
    wa  = 1
    wb  = torch.sqrt(input_derivatives/input_derivatives_plus_one) * wa
    wc  = (lam * wa * input_derivatives + (1-lam) * wb * input_derivatives_plus_one)/input_delta
    ya  = input_cumheights
    yb  = input_heights + input_cumheights
    yc  = ((1-lam) * wa * ya + lam * wb * yb)/((1-lam) * wa + lam * wb)

    if inverse:

        numerator = (lam * wa * (ya - inputs)) * (inputs <= yc).float() \
                  +  ((wc - lam * wb) * inputs + lam * wb * yb - wc * yc) * (inputs > yc).float()

        denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (inputs <= yc).float()\
                    + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

        theta = numerator/denominator

        outputs = theta * input_bin_widths + input_cumwidths

        derivative_numerator = (wa * wc * lam * (yc - ya) * (inputs <= yc).float()\
                             + wb * wc * (1 - lam) * (yb - yc) * (inputs > yc).float())*input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet
    else:

        theta = (inputs - input_cumwidths) / input_bin_widths

        numerator = (wa * ya * (lam - theta) + wc * yc * theta) * (theta <= lam).float()\
                  + (wc * yc * (1 - theta) + wb * yb * (theta - lam)) * (theta > lam).float()

        denominator = (wa * (lam - theta) + wc * theta) * (theta <= lam).float()\
                    + (wc * (1 - theta) + wb * (theta - lam)) * (theta > lam).float()

        outputs = numerator / denominator

        derivative_numerator = (wa * wc * lam * (yc - ya) * (theta <= lam).float()\
                             + wb * wc * (1 - lam) * (yb - yc) * (theta > lam).float())/input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet