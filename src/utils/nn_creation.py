
from matplotlib import cm, pyplot as plt
import torch
import matplotlib.patches as mpatches

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

def visualize_weights(net, title="NN"):
    list_of_list_of_weights = []
    for param in net.parameters():
        list_of_list_of_weights.append(param.data.view(-1).tolist())
    # Visualize as a single bar chart, where each bar is coloured based on which param it appeared in
    # First, flatten the list of lists
    flattened_list_of_weights = [item for sublist in list_of_list_of_weights for item in sublist]
    # Then, create a list of colours, where each colour is repeated the number of times that param appears
    colours = []
    for i, sublist in enumerate(list_of_list_of_weights):
        colours += [cm.hot(i / len(list_of_list_of_weights))] * len(sublist)

    # Produce a legend using module names. -1 to remove final sigmoid module, which has no params
    module_names = [str(m) for m in net.modules()][1:-1] + ['output bias']
    patches = [mpatches.Patch(color=cm.hot(i / len(list_of_list_of_weights)), label=module_names[i]) for i in range(len(module_names))]
    plt.figure(figsize=(6, 8))
    plt.legend(handles=patches)

    plt.title(f"{title}: weight distribution for {len(flattened_list_of_weights)} weights\n(min {round(min(flattened_list_of_weights), sigfigs=3)}, \nmax {round(max(flattened_list_of_weights), sigfigs=3)}, \n|min| {round(min([abs(w) for w in flattened_list_of_weights]), sigfigs=3)})")

    # Finally, plot the weights
    plt.bar(range(len(flattened_list_of_weights)), flattened_list_of_weights, color=colours)        

    plt.savefig(f"{title}_weights.png")
    plt.close()