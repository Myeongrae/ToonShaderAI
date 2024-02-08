import torch

class ImageSmoothLoss(torch.autograd.Function) :
    ''' 
        image smoothness loss
    '''
    @staticmethod
    def forward(ctx, input, target, gamma=10.):
        input_grad_x = (input[:, :, 2:, :] - input[:, :, :-2, :]) / 2
        input_grad_y = (input[:, :, :, 2:] - input[:, :, :, :-2]) / 2
        target_grad_x = (target[:, :, 2:, :] - target[:, :, :-2, :]) / 2 
        target_grad_y = (target[:, :, :, 2:] - target[:, :, :, :-2]) / 2 

        target_grad_x = torch.square(torch.exp(-gamma * target_grad_x))
        target_grad_y = torch.square(torch.exp(-gamma * target_grad_y))

        result = torch.mean(torch.square(input_grad_x)*target_grad_x) + torch.mean(torch.square(input_grad_y)*target_grad_y)
        
        ctx.save_for_backward(input_grad_x, input_grad_y, target_grad_x, target_grad_y)
        return result 

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0] : 
            input_grad_x, input_grad_y, target_grad_x, target_grad_y = ctx.saved_tensors
            b, c, h, w = input_grad_x.size()
            input_grad_x = torch.nn.functional.pad(input_grad_x, (0, 0, 2, 2), 'constant', 0.)
            input_grad_y = torch.nn.functional.pad(input_grad_y, (2, 2), 'constant', 0.)
            target_grad_x = torch.nn.functional.pad(target_grad_x, (0, 0, 2, 2), 'constant', 0.)
            target_grad_y = torch.nn.functional.pad(target_grad_y, (2, 2), 'constant', 0.)

            grad_x = input_grad_x * target_grad_x
            grad_y = input_grad_y * target_grad_y

            grad_input = grad_output * (grad_x[:, :, :-2, :] - grad_x[:, :, 2:, :] + grad_y[:, :, :, :-2] - grad_y[:, :, :, 2:]) / (b*c*h*w)

        return grad_input, None, None
    
def smooth_loss(input, target, gamma=10.) :
    return ImageSmoothLoss.apply(input, target, gamma)

def kl_div_loss(feature) : 
    mean = torch.mean(feature, (-1, -2))
    var = torch.var(feature, (-1, -2))
    loss = -1 + torch.square(mean) + var - torch.log(var + 1e-8)
    return torch.mean(0.5*loss)