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

#We can start with random point. Admittedly here as we have two local minimums where we start does matter, because we can end up in different ones
s_param = torch.randn(1, 1, device=device, dtype=dtype, requires_grad=True)

#Let's use classical learning rate here
learning_rate = 1e-6

#The number of steps is also arbitrary here, it obtained from trial and error
for t in range(1000):

    f = STESum.apply
    loss = torch.sum(f(s_param))
    loss.backward()
    #Update weights using gradient descent
    with torch.no_grad():
        s_param -= learning_rate * s_param.grad
        #Manually zero the gradients after updating weights
        s_param.grad.zero_()

#Print all necessary data
print (data)
print (s_param)
s_one_analytical = torch.tensordot(data, torch.sign(data)) / 100
print (s_one_analytical)
s_two_analytical = torch.multiply(torch.tensordot(data, torch.sign(data)) / 100, -1)
print (s_two_analytical)