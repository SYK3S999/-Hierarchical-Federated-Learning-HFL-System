import torch

def compute_DF(v_c, DS, rho=[1.0, 0.5, 0.01, 1.0, 0.5, 1.0]):
    # Ensure DS is a tensor
    DS = torch.tensor(DS, dtype=torch.float32) if not isinstance(DS, torch.Tensor) else DS
    
    # Convert rho elements to tensors for computation
    rho_tensors = [torch.tensor(r, dtype=torch.float32) for r in rho]
    
    sigma = rho_tensors[3] * torch.exp(-((v_c + rho_tensors[4]) / rho_tensors[5])**2)
    term2 = rho_tensors[0] * torch.exp(-rho_tensors[1] * (rho_tensors[2] * DS))
    return sigma - term2

def aggregate_models(target_model, source_models, weights):
    total_weight = sum(weights)
    for param in target_model.parameters():
        param.data.zero_()
        for model, weight in zip(source_models, weights):
            param.data.add_(model.state_dict()[param.name].data * weight)
        param.data.div_(total_weight + 1e-6)