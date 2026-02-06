import torch
import torch.nn as nn
import torch.nn.functional as F

from ucb_algorithms import UCB_BwK, UCB1, UCB_V, UCB_Tuned, BayesUCB

class ExitBlock(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=1):
        super(ExitBlock, self).__init__()
        layers = []
        channels = in_channels

        for _ in range(num_convs):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)

class ExitBlock50(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=1, reduction=0.25):
        super(ExitBlock50, self).__init__()

        reduced_channels = max(16, int(in_channels * reduction))
        layers = []

        if num_convs > 0:
            layers.append(nn.Conv2d(in_channels, reduced_channels, kernel_size=1, stride=1))
            layers.append(nn.BatchNorm2d(reduced_channels))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(num_convs):
                layers.append(nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(reduced_channels))
                layers.append(nn.ReLU(inplace=True))

            self.features = nn.Sequential(*layers)
            classifier_in = reduced_channels
        else:
            self.features = nn.Identity()
            classifier_in = in_channels

        self.classifier = nn.Linear(classifier_in, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)

class ResNetEE18(nn.Module):
    def __init__(self, block, layers, num_classes=10, confidence_threshold=0.9):
        super(ResNetEE18, self).__init__()
        self.inplanes = 64
        self.confidence_threshold = confidence_threshold

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.exit0 = ExitBlock(64, num_classes, num_convs=3)
        self.exit1 = ExitBlock(128, num_classes, num_convs=2)
        self.exit2 = ExitBlock(256, num_classes, num_convs=1)


        self.early_exits = [self.exit0, self.exit1, self.exit2]
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, exit_layer=None):
        if self.training:
            x = self.conv1(x)
            x = self.maxpool(x)

            x0 = self.layer0(x)
            out0 = self.exit0(x0)

            x1 = self.layer1(x0)
            out1 = self.exit1(x1)

            x2 = self.layer2(x1)
            out2 = self.exit2(x2)

            x3 = self.layer3(x2)

            xf = self.avgpool(x3)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)

            return [out0, out1, out2, out_final]

        else:

          x = self.conv1(x)
          x = self.maxpool(x)
          x = self.layer0(x)
          out0 = self.exit0(x)
          if self._confident_enough(out0):
            return out0, 0

          x = self.layer1(x)
          out1 = self.exit1(x)
          if self._confident_enough(out1):
            return out1, 1

          x = self.layer2(x)
          out2 = self.exit2(x)
          if self._confident_enough(out2):
            return out2, 2

          x = self.layer3(x)

          xf = self.avgpool(x)
          xf = torch.flatten(xf, 1)
          out_final = self.fc(xf)
          return out_final, 3

    def _confident_enough(self, logits):
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return conf.item() >= self.confidence_threshold

class ResNetEE50(nn.Module):
    def __init__(self, block, layers, num_classes=10, confidence_threshold=0.9):
        super(ResNetEE50, self).__init__()
        self.inplanes = 64
        self.confidence_threshold = confidence_threshold

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.exit0 = ExitBlock50(64 * block.expansion, num_classes, num_convs=3)
        self.exit1 = ExitBlock50(128 * block.expansion, num_classes, num_convs=2)
        self.exit2 = ExitBlock50(256 * block.expansion, num_classes, num_convs=1)

        self.early_exits = [self.exit0, self.exit1, self.exit2]
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, exit_layer=None):
        if self.training:
            x = self.conv1(x)
            x = self.maxpool(x)

            x0 = self.layer0(x)
            out0 = self.exit0(x0)

            x1 = self.layer1(x0)
            out1 = self.exit1(x1)

            x2 = self.layer2(x1)
            out2 = self.exit2(x2)

            x3 = self.layer3(x2)

            xf = self.avgpool(x3)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)

            return [out0, out1, out2, out_final]

        else:

          x = self.conv1(x)
          x = self.maxpool(x)
          x = self.layer0(x)
          out0 = self.exit0(x)
          if self._confident_enough(out0):
            return out0, 0

          x = self.layer1(x)
          out1 = self.exit1(x)
          if self._confident_enough(out1):
            return out1, 1

          x = self.layer2(x)
          out2 = self.exit2(x)
          if self._confident_enough(out2):
            return out2, 2

          x = self.layer3(x)

          xf = self.avgpool(x)
          xf = torch.flatten(xf, 1)
          out_final = self.fc(xf)
          return out_final, 3

    def _confident_enough(self, logits):
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return conf.item() >= self.confidence_threshold

class Gmodel(nn.Module):

  def __init__(self, hidden_dim):
        super(Gmodel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

  def forward(self, x):
        return self.net(x).squeeze(-1)

class ResNetEE18U(nn.Module):
    def __init__(self, block, layers, num_classes=10, confidence_threshold=0.9, arms=[], bwk=False, mode=None):
        super(ResNetEE18U, self).__init__()
        self.inplanes = 64
        self.confidence_threshold = confidence_threshold

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.exit0 = ExitBlock(64, num_classes, num_convs=3)
        self.exit1 = ExitBlock(128, num_classes, num_convs=2)
        self.exit2 = ExitBlock(256, num_classes, num_convs=1)


        self.early_exits = [self.exit0, self.exit1, self.exit2]
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]

        self.thresholds = arms
        self.bwk = bwk
        self.mode = mode
        self.gating=None

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

    def select_tau(self):
        self.current_tau = self.ucb.select_arm()
        return self.current_tau

    def update_ucb(self, logits, exit_idx):
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        pro = self.gating(logits)

        if self.bwk == False:
          reward = conf*(1-pro) - 0.0025 * exit_idx
          self.ucb.update(self.current_tau, reward)

        if self.bwk == True:
          reward = conf*(1-pro)
          self.ucb.update(self.current_tau, reward=reward, actual_cost=0.0025*exit_idx)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, exit_layer=None):
        if self.training:
            x = self.conv1(x)
            x = self.maxpool(x)

            x0 = self.layer0(x)
            out0 = self.exit0(x0)

            x1 = self.layer1(x0)
            out1 = self.exit1(x1)

            x2 = self.layer2(x1)
            out2 = self.exit2(x2)

            x3 = self.layer3(x2)

            xf = self.avgpool(x3)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)

            return [out0, out1, out2, out_final]

        else:

          tau = self.select_tau()

          x = self.conv1(x)
          x = self.maxpool(x)
          x = self.layer0(x)
          out0 = self.exit0(x)
          prob = F.softmax(out0, dim=1)
          con, _ = prob.max(dim=1)
          if con >= tau:
            self.update_ucb(out0, 0)
            return out0, 0

          x = self.layer1(x)
          out1 = self.exit1(x)
          prob = F.softmax(out1, dim=1)
          con, _ = prob.max(dim=1)
          if con >= tau:
            self.update_ucb(out1, 1)
            return out1, 1

          x = self.layer2(x)
          out2 = self.exit2(x)
          prob = F.softmax(out2, dim=1)
          con, _ = prob.max(dim=1)
          if con >= tau:
            self.update_ucb(out2, 2)
            return out2, 2

          x = self.layer3(x)

          xf = self.avgpool(x)
          xf = torch.flatten(xf, 1)
          out_final = self.fc(xf)
          return out_final, 3

    def _confident_enough(self, logits):
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return conf.item() >= self.confidence_threshold

class ResNetEE50U(nn.Module):
    def __init__(self, block, layers, num_classes=10, confidence_threshold=0.9, arms=[], bwk=False, mode=None):
        super(ResNetEE50U, self).__init__()
        self.inplanes = 64
        self.confidence_threshold = confidence_threshold

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.exit0 = ExitBlock50(64 * block.expansion, num_classes, num_convs=3)
        self.exit1 = ExitBlock50(128 * block.expansion, num_classes, num_convs=2)
        self.exit2 = ExitBlock50(256 * block.expansion, num_classes, num_convs=1)

        self.early_exits = [self.exit0, self.exit1, self.exit2]
        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]

        self.thresholds = arms
        self.bwk = bwk
        self.mode = mode
        self.gating=None

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

    def select_tau(self):
        self.current_tau = self.ucb.select_arm()
        return self.current_tau

    def update_ucb(self, logits, exit_idx):
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        pro = self.gating(logits)

        if self.bwk == False:
          reward = conf*(1-pro) - 0.0025 * exit_idx
          self.ucb.update(self.current_tau, reward)

        if self.bwk == True:
          reward = conf*(1-pro)
          self.ucb.update(self.current_tau, reward=reward, actual_cost=0.0025*exit_idx)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, exit_layer=None):
        if self.training:
            x = self.conv1(x)
            x = self.maxpool(x)

            x0 = self.layer0(x)
            out0 = self.exit0(x0)

            x1 = self.layer1(x0)
            out1 = self.exit1(x1)

            x2 = self.layer2(x1)
            out2 = self.exit2(x2)

            x3 = self.layer3(x2)
            xf = self.avgpool(x3)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)

            return [out0, out1, out2, out_final]

        else:

            tau = self.select_tau()

            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.layer0(x)
            out0 = self.exit0(x)
            prob = F.softmax(out0, dim=1)
            con, _ = prob.max(dim=1)
            if con >= tau:
                self.update_ucb(out0, 0)
                return out0, 0

            x = self.layer1(x)
            out1 = self.exit1(x)
            prob = F.softmax(out1, dim=1)
            con, _ = prob.max(dim=1)
            if con >= tau:
                self.update_ucb(out1, 1)
                return out1, 1

            x = self.layer2(x)
            out2 = self.exit2(x)
            prob = F.softmax(out2, dim=1)
            con, _ = prob.max(dim=1)
            if con >= tau:
                self.update_ucb(out2, 2)
                return out2, 2

            x = self.layer3(x)
            xf = self.avgpool(x)
            xf = torch.flatten(xf, 1)
            out_final = self.fc(xf)
            return out_final, 3

    def _confident_enough(self, logits):
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return conf.item() >= self.confidence_threshold

