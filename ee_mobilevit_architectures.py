import torch.nn as nn
import torch.nn.functional as F

from ucb_algorithms import UCB_BwK, UCB1, UCB_V, UCB_Tuned

class EarlyExitHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.exit = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(in_channels // 2, num_classes)

    def forward(self, x):
        x = self.exit(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MobileViTWithEarlyExits(nn.Module):
    def __init__(self, base_model, exit_points=['mvit_0', 'mvit_1'], num_classes=100, exit_threshold=0.9):
        super().__init__()
        self.base = base_model
        self.exit_threshold = exit_threshold
        self.num_classes = num_classes

        self.exit_channels = {
            'mvit_0': base_model.mvit[0].conv4[0].out_channels,
            'mvit_1': base_model.mvit[1].conv4[0].out_channels
        }

        self.exits = nn.ModuleDict({
            name: EarlyExitHead(self.exit_channels[name], num_classes)
            for name in exit_points
        })

    def forward(self, x, targets=None, train_mode=True):
        losses = []

        x = self.base.conv1(x)
        x = self.base.mv2[0](x)
        x = self.base.mv2[1](x)
        x = self.base.mv2[2](x)
        x = self.base.mv2[3](x)
        x = self.base.mv2[4](x)

        x = self.base.mvit[0](x)
        if 'mvit_0' in self.exits:
            out = self.exits['mvit_0'](x)
            if train_mode:
                losses.append(out)
            else:
                conf, pred = F.softmax(out, dim=1).max(dim=1)
                if conf.mean() > self.exit_threshold:
                    return 0, pred

        x = self.base.mv2[5](x)
        x = self.base.mvit[1](x)
        if 'mvit_1' in self.exits:
            out = self.exits['mvit_1'](x)
            if train_mode:
                losses.append(out)
            else:
                conf, pred = F.softmax(out, dim=1).max(dim=1)
                if conf.mean() > self.exit_threshold:
                    return 1, pred

        x = self.base.mv2[6](x)
        x = self.base.mvit[2](x)
        x = self.base.conv2(x)
        x = self.base.pool(x).view(-1, x.shape[1])
        out = self.base.fc(x)

        if train_mode:
            losses.append(out)
            return losses
        else:
            _, pred = F.softmax(out, dim=1).max(dim=1)
            return len(self.exits), pred

class MobileViTWithUCB(nn.Module):
    def __init__(self, base_model, exit_points=['mvit_0', 'mvit_1'], num_classes=100,
                 arms=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85], bwk=False, mode=None):

        super().__init__()
        self.base = base_model
        self.num_classes = num_classes

        self.thresholds = arms
        self.mode = mode
        self.bwk = bwk
        self.gating=None
        self.tau = None

        if self.mode == "ucb1":
          self.ucb = UCB1(self.thresholds)
        if self.mode == "tuned":
          self.ucb = UCB_Tuned(self.thresholds)
        if self.mode == "bwk":
          self.ucb = UCB_BwK(self.thresholds)
        if self.mode == "v":
          self.ucb = UCB_V(self.thresholds)
        if self.mode == "bayes":
          self.ucb = BayesUCB(self.thresholds)

        self.current_tau = 0.0
        self.num_early_exits = len(exit_points)
        self.cost_per_exit = 0.0033

        self.exit_channels = {
            'mvit_0': base_model.mvit[0].conv4[0].out_channels,
            'mvit_1': base_model.mvit[1].conv4[0].out_channels
        }

        self.exits = nn.ModuleDict({
            name: EarlyExitHead(self.exit_channels[name], num_classes)
            for name in exit_points
        })

    def select_tau(self):
        self.current_tau = self.ucb.select_arm()
        return self.current_tau

    def update_ucb(self, logits, exit_idx):

        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        pro = self.gating(logits)

        if self.bwk == True:
          reward = (conf * (1 - pro))
          self.ucb.update(self.current_tau, reward, (self.cost_per_exit * exit_idx))

        else:

          reward = (conf * (1 - pro)) - self.cost_per_exit * exit_idx
          self.ucb.update(self.current_tau, reward)


    def forward(self, x, targets=None, train_mode=True):
        losses = []

        x = self.base.conv1(x)
        x = self.base.mv2[0](x)
        x = self.base.mv2[1](x)
        x = self.base.mv2[2](x)
        x = self.base.mv2[3](x)
        x = self.base.mv2[4](x)

        if not train_mode:
            tau = self.select_tau()

            self.tau = tau

        x = self.base.mvit[0](x)
        if 'mvit_0' in self.exits:
            out0 = self.exits['mvit_0'](x)

            if train_mode:
                losses.append(out0)
            else:
                prob = F.softmax(out0, dim=1)
                con, pred = prob.max(dim=1)

                if con >= tau:
                    self.update_ucb(out0, 0)
                    return 0, pred

        x = self.base.mv2[5](x)
        x = self.base.mvit[1](x)
        if 'mvit_1' in self.exits:
            out1 = self.exits['mvit_1'](x)

            if train_mode:
                losses.append(out1)
            else:
                prob = F.softmax(out1, dim=1)
                con, pred = prob.max(dim=1)
                if con >= tau:
                    self.update_ucb(out1, 1)
                    return 1, pred

        x = self.base.mv2[6](x)
        x = self.base.mvit[2](x)
        x = self.base.conv2(x)
        x = self.base.pool(x).view(-1, x.shape[1])
        out_final = self.base.fc(x)

        if train_mode:
            losses.append(out_final)
            return losses
        else:
            final_exit_idx = self.num_early_exits
            self.update_ucb(out_final, final_exit_idx)

            _, pred = F.softmax(out_final, dim=1).max(dim=1)
            return final_exit_idx, pred