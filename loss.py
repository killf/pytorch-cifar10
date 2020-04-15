import torch
import numpy as np


class CycleLR:
    def __init__(self, optimizer, min_lr=1e-3, max_lr=1., step_size=5, repeat_times=7, final_lr=1e-5, final_step=10,
                 last_epoch=-1):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.repeat_times = repeat_times
        self.final_lr = final_lr
        self.final_step = final_step
        self.last_epoch = last_epoch

    @property
    def epochs(self):
        return self.step_size * 2 + self.final_step

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        lr = None
        if self.last_epoch < self.step_size * 2 and self.step_size > 1:
            a, b = divmod(self.last_epoch, self.step_size)
            if a // 2 == 1:
                b = self.step_size - b

            lr = self.min_lr + (self.max_lr - self.min_lr) / (self.step_size - 1) * b
        elif self.last_epoch < self.epochs and self.final_step > 1:
            b = self.epochs - self.last_epoch
            lr = self.final_lr + (self.min_lr - self.final_lr) / (self.final_step - 1) * b

        if lr is None:
            return

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def find_lr(data_loader, net, criterion, device, min_lr=1e-5, max_lr=10, gamma=1.1):
    lr_ls, loss_ls = [], []
    origin_model = net.state_dict()

    lr = min_lr
    net.train()
    for step, (x, y_true) in enumerate(data_loader):
        if lr > max_lr:
            break

        net.load_state_dict(origin_model)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)

        x, y_true = x.to(device), y_true.to(device)

        y_pred = net(x)
        loss = criterion(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = net(x)
        loss = criterion(y_pred, y_true)

        lr_ls.append(lr)
        loss_ls.append(loss.item())

        lr = lr * gamma

    net.load_state_dict(origin_model)
    idx = np.argmin(loss_ls)
    return lr_ls[idx]
