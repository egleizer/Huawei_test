import torch

class STETanh(torch.autograd.Function):

    def forward(ctx, input):
        return torch.sign(input)

    def backward(ctx, grad_output):
        tanh = torch.tanh(grad_output)
        return (1 - torch.pow(tanh, 2))