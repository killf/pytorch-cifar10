import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=4, reduction="mean", numeric_stable_mode=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.numeric_stable_mode = numeric_stable_mode

    def forward(self, inputs, targets):
        gt = F.one_hot(targets, num_classes=inputs.size(1))
        pred = F.softmax(inputs, dim=-1)
        if self.numeric_stable_mode:
            pred = torch.clamp(pred, min=1e-7, max=1. - 1e-7)

        loss = -self.alpha * gt * torch.log(pred) * torch.pow(1. - pred, self.gamma)
        loss = torch.sum(loss, dim=-1)

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class OHEMLoss(nn.Module):
    def __init__(self, reduction="mean", numeric_stable_mode=True):
        super(OHEMLoss, self).__init__()
        self.reduction = reduction
        self.numeric_stable_mode = numeric_stable_mode

    def forward(self, inputs, targets):
        gt = F.one_hot(targets, num_classes=inputs.size(1))
        pred = F.softmax(inputs, dim=-1)
        if self.numeric_stable_mode:
            pred = torch.clamp(pred, min=1e-7, max=1. - 1e-7)

        loss = -1 * gt * torch.log(pred)
        loss = torch.sum(loss, dim=-1)

        pred = torch.argmax(pred, dim=-1)
        weight = (pred != targets).type_as(loss)
        # weight = torch.clamp(weight, min=0.3, max=0.7)
        weight = weight + 1
        loss = weight * loss

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


if __name__ == '__main__':
    a = torch.from_numpy(np.array([1, 3, 3], dtype=np.int64))
    b = torch.from_numpy(np.array([[0.7, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0],
                                   [0.0, 0.0, 0.3, 0.4, 0.3, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.2, 0.7, 0.1, 0.0, 0.0, 0.0]], dtype=np.float32))

    loss1 = F.cross_entropy(b, a)
    print(loss1)

    loss2 = FocalLoss(gamma=4)(b, a)
    print(loss2)

    loss3 = OHEMLoss()(b, a)
    print(loss3)
