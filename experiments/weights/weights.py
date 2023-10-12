import json
import numpy as np
from numpy import linalg as LA

PATH = "/nobackup/naman/LRS_NF/experiments/weights/"
dataset = PATH + "/cifar-10"

g = dataset + "-g.json"
autoaug_augmix_h_ft_on_g = dataset + "-autoaug-augmix-h-ft-on-g.json"
augmix_h_ft_on_g = dataset + "-augmix-h-ft-on-g.json"

f1 = open(g)
g = json.load(f1)
f1.close()

f2 = open(autoaug_augmix_h_ft_on_g)
autoaug_augmix_h_ft_on_g = json.load(f2)
f2.close()

f3 = open(augmix_h_ft_on_g)
augmix_h_ft_on_g = json.load(f3)
f3.close()

# layers
tune_layers = [
    "_transform._transforms.1._transforms.2._transforms.7",
    "_transform._transforms.1._transforms.2._transforms.8"
]


for k, v in g.items():
    v1 = np.array(v)
    v2 = np.array(augmix_h_ft_on_g[k])
    v3 = np.array(autoaug_augmix_h_ft_on_g[k])
    # print("Means")
    # print(k, np.mean(v1), np.mean(v2), np.mean(v3))

    print("max abs diff")
    max_abs_diff_v3 = np.max(np.abs(v3 - v1)).mean()
    max_abs_diff_v2 = np.max(np.abs(v2 - v1)).mean()
    
    print(k)
    print("autoaug_augmix_h_ft_on_g and g", max_abs_diff_v2)
    print("augmix_h_ft_on_g and g", max_abs_diff_v3)
    print("")

# TODO: should we do take a mean over all the sub-layers for each tune_layer?
