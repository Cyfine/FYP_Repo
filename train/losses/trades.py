import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class TradesLoss:
    def __init__(self,
                 model,
                 device,
                 optimizer,
                 step_size,
                 epsilon,
                 perturb_steps,
                 beta, clip_min,
                 clip_max,
                 distance,
                 natural_criterion=nn.CrossEntropyLoss()):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.distance = distance
        self.natural_criterion = natural_criterion

        pass

    def __call__(self, x_natural, y):
        criterion_kl = nn.KLDivLoss(size_average=False)
        self.model.eval()
        batch_size = len(x_natural)
        # generate adversarial example
        x_adv = (
                x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        )
        if self.distance == "l_inf":
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(
                        F.log_softmax(self.model(x_adv), dim=1),
                        F.softmax(self.model(x_natural), dim=1),
                    )
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(
                    torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon
                )
                x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        elif self.distance == "l_2":
            delta = 0.001 * torch.randn(x_natural.shape).to(self.device).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD([delta], lr=self.epsilon / self.perturb_steps * 2)

            for _ in range(self.perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(
                        F.log_softmax( self.model(adv), dim=1), F.softmax(self.model(x_natural), dim=1)
                    )
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(
                        delta.grad[grad_norms == 0]
                    )
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(self.clip_min, self.clip_max).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_adv = Variable(x_natural + delta, requires_grad=False)
        else:
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, self.clip_min, self.clip_max), requires_grad=False)
        # zero gradient
        self.optimizer.zero_grad()
        # calculate robust loss
        logits = self.model(x_natural)
        loss_natural = self.natural_criterion(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(
            F.log_softmax(self.model(x_adv), dim=1), F.softmax(self.model(x_natural), dim=1)
        )
        loss = loss_natural + self.beta * loss_robust
        return loss



