import torch

def compute_DF(v_c, DS, rho=[1.0, 0.5, 0.01, 1.0, 0.5, 1.0]):
    sigma = rho[3] * torch.exp(-((v_c + rho[4]) / rho[5])**2)
    term2 = rho[0] * torch.exp(-rho[1] * (rho[2] * DS))
    return sigma - term2

def aggregate_models(target_model, source_models, weights):
    total_weight = sum(weights)
    for param in target_model.parameters():
        param.data.zero_()
        for model, weight in zip(source_models, weights):
            param.data.add_(model.state_dict()[param.name].data * weight)
        param.data.div_(total_weight + 1e-6)