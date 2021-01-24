import torch

class STESum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        inside = torch.pow(torch.div(data, input) - torch.sign(torch.div(data, input)),2)
        answer = (input**2) * torch.sum(inside)
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        answer = 100 * grad_output - torch.multiply(input , torch.sign(torch.div(input, grad_input)))
        answer = 2 * answer
        return (answer)

dtype = torch.float
device = torch.device("cpu")

data = torch.normal(0, 1, size=(1, 100), device=device, dtype=dtype, requires_grad=True)
#we can start with any point. Admittedly here as we have two local minimums where we start does matter, because we can end up in different ones
s_param = torch.randn(1, 1, device=device, dtype=dtype, requires_grad=True)
#let's use classical learning rate here
learning_rate = 1e-6
#the number of steps is also arbitrary here, taken from torch documentation
for t in range(500):

    f = STESum.apply
    loss = torch.sum(f(s_param))

    # To apply our Function, we use Function.apply method
    #function = STESum.apply(s_param)

    # Forward pass
    #x.mm(w1)
    #y_pred = function()

    # Compute and print loss
   # loss = y_pred.clone().detach()
   # loss.requires_grad = True
    #loss = torch.sum(function)

    # Use autograd to compute the backward pass.
    loss.backward()
    # # Update weights using gradient descent
    #print(s_param.grad)
    with torch.no_grad():
        s_param -= learning_rate * s_param.grad
        # Manually zero the gradients after updating weights
        s_param.grad.zero_()

print (s_param)