# Invertible Generative Modeling using Linear Rational Splines

This repository includes the Python code of [the paper](https://arxiv.org/abs/2001.05168):

> H. M. Dolatabadi, S. Erfani, and C. Leckie, "Invertible Generative Modeling using Linear Rational Splines," in _The 23rd International Conference on Artifcial Intelligence and Statistics (AISTATS) 2020_, 3-5 June 2020, Palermo, Sicily, Italy.

The code is mostly taken from [Neural Spline Flows](https://github.com/bayesiains/nsf) repository, initially downloaded on mid August 2019.

## Running the code

To run experiments of the paper, take the following steps:

1. Install the dependencies using "./environment.yml" by running:
```bash
conda env create -f environment.yml
```

2. Set the path variables in "./experiments/cutils/io.py" accordingly.

3. Download the raw tabular data (Power, Gas, HEPMASS, and MiniBooNE + BSDS300) from https://zenodo.org/record/1161203#.Wmtf_XVl8eN provided by Papamakarios et al. for Masked Autoregressive Flows. Extract the data into the root "./data/".

4. Set the "DATAROOT" environment variable to the dataroot folder, namely "./data/".

5. Run "bsds300.py", "gas.py", "hepmass.py", "miniboone.py" and "power.py" one after another to extract and split the data accordingly.

6. Now, you can run each experiment of the paper by running its associated experiment:

### Plane
```bash
python ./experiments/plane.py --dataset_name=rings --base_transform_type=rl-coupling
```

### Face
```bash
python ./experiments/face.py --dataset_name=einstein --base_transform_type=rl
```

### Tabular Data
```bash
python ./experiments/uci.py --dataset_name=bsds300 --base_transform_type=rl-coupling --tail_bound=3 --num_bins=8
```

### Image Generation
First set the config on the associated .json file in "./experiments/image_configs/". Then run:
```bash
python experiments/images.py with experiments/image_configs/cifar-10-8bit-RL.json
```

To compute the BPD on the test data, run:
```bash
python experiments/images.py eval_on_test with experiments/image_configs/cifar-10-8bit-RL.json flow_checkpoint='<saved_checkpoint>'
```

### VAE
```bash
python experiments/vae.py --prior_type=rl-coupling --approximate_posterior_type=rl-coupling
```

## Citation
```bash
@inproceedings{dolatabadi2020lrs,
  title={Invertible Generative Modeling using Linear Rational Splines},
  author={M. Dolatabadi, Hadi and Erfani, Sarah and Christopher, Leckie},
  booktitle={The 23rd International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2020}
}
```
