import torch

class STELinear(torch.autograd.Function):

    def forward(ctx, input):
        return torch.sign(input)

    def backward(ctx, grad_output):
        return torch.ones_like(grad_output)