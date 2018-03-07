import torch
from torch.optim.optimizer import Optimizer


def _matrix_power(matrix, power):
    u, s, v = torch.svd(matrix)
    return u @ s.pow_(power).diag() @ v.t()


class Shampoo(Optimizer):

    def __init__(self, params, lr=1e-1, momentum=0, epsilon=1e-4, update_freq=1):
        """

        :param params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr (float, optional): learning rate (default: 1e-1)
        :param momentum: (float, optional): momentum factor (default: 0)
        :param epsilon: (float, optional): momentum factor (default: 1e-4)
        :param update_freq: (int, optional): update frequency to compute inverse (default: 1)
        """
        defaults = dict(lr=lr, momentum=momentum, epsilon=epsilon, update_freq=update_freq)
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                momentum = group["momentum"]
                ndim = p.ndimension()
                if ndim > 1:
                    original_size = p.size()
                    if ndim > 2:
                        grad = grad.view(original_size[0], -1)
                    _m, _n = grad.size()
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        state["L"] = group["epsilon"] * torch.eye(_m, out=grad.new(_m, _m))
                        state["R"] = group["epsilon"] * torch.eye(_n, out=grad.new(_n, _n))
                        state["quarter_L"] = None
                        state["quarter_R"] = None
                        if momentum > 0:
                            state["exp_avg"] = grad.new(_m, _n)

                    if momentum > 0:
                        state["exp_avg"].mul_(momentum).add_(momentum, grad)
                        grad = state["exp_avg"]

                    grad_t = grad.t()
                    state["L"].add_(grad @ grad_t)
                    state["R"].add_(grad_t @ grad)

                    if state["step"] % group["update_freq"] == 0:
                        state["quarter_L"] = _matrix_power(state["L"], -0.25)
                        state["quarter_R"] = _matrix_power(state["R"], -0.25)
                    grad = (state["quarter_L"] @ grad @ state["quarter_R"]).view(original_size)
                    state["step"] += 1

                p.data.add_(-group["lr"], grad)

        return loss
