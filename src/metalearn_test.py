import torch
import torch.nn as nn


if __name__ == "__main__":
    
    # generate w
    pis_w = torch.nn.Parameter(torch.rand(1))
    w = torch.ones(2) * pis_w
    w.retain_grad()

    # set w into network
    ff_net = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
    param_counts = [torch.tensor(m.shape).prod().item() for m in ff_net.parameters()]
    param_counts = tuple(map(int, param_counts))
    split_params = torch.split(w, param_counts)
    #for w, p in zip(split_params, ff_net.parameters()):
    #    p.data = w.view(p.shape)
    w_split_index = 0
    for module_key in ff_net._modules:
        m = ff_net._modules[module_key]
        for param_key in m._parameters:
            p = m._parameters[param_key]
            m._parameters[param_key] = w[w_split_index:w_split_index+p.numel()].view(p.shape)
            w_split_index += p.numel()


    # forward through network
    x = torch.tensor([7.0])
    y = ff_net(x)

    # output.backward()
    y.backward()

    # gradient of weights that produced attempted weights
    print(pis_w.grad)
    # gradient of attempted weights
    print(w.grad)
    print([p.grad for p in ff_net.parameters()])