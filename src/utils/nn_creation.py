
from matplotlib import cm, pyplot as plt
import torch
import matplotlib.patches as mpatches
from sigfig import round

def set_params(net, params):
    # set w into network
    #for w, p in zip(split_params, net.parameters()):
    #    p.data = w.view(p.shape)
    w_split_index = 0
    for module_key in net._modules:
        m = net._modules[module_key]
        for param_key in m._parameters:
            p = m._parameters[param_key]
            m._parameters[param_key] = params[w_split_index:w_split_index+p.numel()].view(p.shape)
            w_split_index += p.numel()

def flat_parameters(net):
    # Return all the parameters of the network as a single 1D tensor, preserving grads
    # TODO: check that this is the same order/shape as set_params!
    # check whether net is already a parameter
    if isinstance(net, torch.nn.parameter.Parameter):
        return net
    else:
        return torch.cat([p.view(-1) for p in net.parameters()])

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from torch.nn import Module

def replace_multi(str, replacements):
    for k, v in replacements.items():
        str = str.replace(k, v)
    return str

def visualize_weights(net: Module, title: str = "NN", dir: str = "."):
    flattened_weights = []
    colours = []
    patches = []

    # Exclude the network itself
    modules = list(net.named_modules())[1:]
    num_modules = len(modules)

    # Iterate over named modules
    for i, (name, module) in enumerate(modules):
        # Access parameters of the current module
        for j, (param_name, param) in enumerate(module.named_parameters()):
            param_weights = param.data.view(-1).tolist()
            
            if param_weights:  # Only append if there are any weights
                flattened_weights.extend(param_weights)
                # Check if the parameter is a weight or a bias
                is_bias = 'bias' in param_name
                alpha = 0.5 if is_bias else 1.0  # less saturation for biases
                colours.extend([cm.hot(i / num_modules, alpha=alpha)] * len(param_weights))

                module_name_formatted = replace_multi(
                    str(module),
                    {
                        ", bias=True": "",
                        ", bias=False": "",
                        "in_features=": "in=",
                        "out_features=": "out="
                    }
                )
                #module_name_formatted = str(module).replace(", bias=True", "")
                #module_name_formatted = module_name_formatted.replace(", bias=False", "")
                patches.append(mpatches.Patch(color=cm.hot(i / num_modules, alpha=alpha), label=f"{module_name_formatted} ({param_name})"))

    plt.figure(figsize=(6, 8))
    plt.legend(handles=patches, loc='upper right')

    plt.title(f"{title}: values of {len(flattened_weights)} parameters\n($|$max$|$={round(max([abs(w) for w in flattened_weights]), 2)}, $|$min$|$={round(min([abs(w) for w in flattened_weights]), 2)})")

    # Plot the weights
    plt.bar(range(len(flattened_weights)), flattened_weights, color=colours)        

    # Add transparent horizontal grid lines
    plt.grid(axis='y', alpha=0.5)

    plt.savefig(f"{dir}/{title}_weights.pdf")
    plt.close()

