import argparse
import os
import pandas as pd

from data.dataloader10 import data_loader10
from data.dataloader100 import data_loader100

from train.mobilevit_base import *
from train.mobilevit_ee_static import *
from train.mobilevit_ee_dynamic import *

from train.resnet_base import *
from train.resnet_ee_static  import *
from train.resnet_ee_dynamic import *

from codecarbon import EmissionsTracker

def evaluation(model, test_loader, model_name, early=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    correct = 0
    total = 0
    NUM_EXITS = 4
    exit_counts = [0] * NUM_EXITS if early else None

    start_time = time.time()
    tracker = EmissionsTracker(measure_power_secs=1)
    tracker.start()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if early:
                preds, exit_id = model(images)
                exit_counts[exit_id] += 1
            else:
                preds = model(images)

            predicted = preds.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    tracker.stop()
    end_time = time.time()

    accuracy = 100 * correct / total
    inference_time = end_time - start_time

    data = tracker._prepare_emissions_data()
    energy = data.energy_consumed

    result = {
        "model_name": model_name,
        "accuracy": accuracy,
        "inference_time_sec": inference_time,
        "energy_kWh": energy,
    }

    for i in range(4):
        if early:
            result[f"exit_{i}_count"] = exit_counts[i]
        else:
            result[f"exit_{i}_count"] = float('nan')

    return result

class Evaluate:

    def __init__(self, model, gating_model, dest, resnet, mvit, variant, src, ucb, hundred, ten):

        if not os.path.exists(dest):
            os.makedirs(dest)
        if not os.path.exists(model) or not os.path.exists(gating_model):
            raise ValueError("Model or Gating model does not exist")
        if not os.path.exists(src):
            raise ValueError("Backbone for mvit is not available")

        self.model = model
        self.gating_model = gating_model

        self.dest = dest

        self.resnet = resnet
        self.mvit = mvit

        self.variant = variant

        self.src = src
        self.ucb = ucb

        self.hundred = hundred
        self.ten = ten

        self.test_loader = None
        self.arms = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        self.num_classes = None

        if self.hundred:
            self.num_classes = 100
        elif self.ten:
            self.num_classes = 10

        return

    def testing_data(self):
        if self.ten:
            self.test_loader = data_loader10(data_dir=self.dest + '/data',
                                                     batch_size=1, test=True)
        else:
            self.test_loader = data_loader100(data_dir=self.dest + '/data',
                                                     batch_size=1, test=True)
        return

    def evaluation(self):

        model = None

        if self.resnet:
            if self.variant == 50:
                if self.ucb != "bwk":
                    model = ResNetEE50U(ResidualBlock50, layers=[3, 4, 6, 3], num_classes=10, mode=self.ucb,
                                        arms=self.arms)
                else:
                    model = ResNetEE50U(ResidualBlock50, layers=[3, 4, 6, 3], num_classes=10, mode=self.ucb,
                                        arms=self.arms, bwk=True)
            elif self.variant == 18:
                if self.ucb != "bwk":
                    model = ResNetEE18U(ResidualBlock, layers=[2, 2, 2, 2], num_classes=10, mode=self.ucb,
                                        arms=self.arms)
                else:
                    model = ResNetEE18U(ResidualBlock, layers=[2, 2, 2, 2], num_classes=10, mode=self.ucb,
                                        arms=self.arms, bwk=True)
            else:
                if self.ucb != "bwk":
                    model = ResNetEE18U(ResidualBlock, layers=[3, 4, 6, 3], num_classes=10, mode=self.ucb,
                                        arms=self.arms)
                else:
                    model = ResNetEE18U(ResidualBlock, layers=[3, 4, 6, 3], num_classes=10, mode=self.ucb,
                                        arms=self.arms, bwk=True)

            model.load_state_dict(torch.load(self.model))

        if self.mvit:

            pretrained_model = mobilevit_xxs()
            pretrained_model.load_state_dict(torch.load(self.src))

            if self.ucb != "bwk":
                model = MobileViTWithUCB(base_model=pretrained_model, exit_points=['mvit_0', 'mvit_1'],
                                           num_classes=self.num_classes, arms=self.arms, bwk=False, mode=self.ucb)
            else:
                model = MobileViTWithUCB(base_model=pretrained_model, exit_points=['mvit_0', 'mvit_1'],
                                           num_classes=self.num_classes, arms=self.arms, bwk=True, mode=self.ucb)

            model.load_state_dict(torch.load(self.model))

        gating = Gmodel(hidden_dim=self.num_classes)
        gating.load_state_dict(torch.load(self.gating_model))

        model.gating = gating

        data = evaluation(model=model, test_loader=self.test_loader, model_name=self.model, early=True)

        df = pd.DataFrame(data)

        df.to_csv(self.dest + '/data.csv', index=False)

        return

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dest", type=str, required=True, default=None, help="Where to store results?")
    parser.add_argument("--model", type=str, required=True, default=None,
                        help="Where to look for the pretrained model?")
    parser.add_argument("--gating_model", type=str, required=True, default=None,
                        help="Where to look for the pretrained gating model?")
    parser.add_argument("--ucb", choices=["ucb1", "tuned", "bwk", "v", "bayes"],
                        help="choose ucb algorithm")
    parser.add_argument("--resnet", type=bool, required=False, default=False,
                        help="train a resnet variant")
    parser.add_argument("--mvit", type=bool, required=False, default=False, help="train a mvit variant")

    # if resnet is True
    parser.add_argument("--variant", type=int, required=False, default=None,
                        help="What is the variant of ResNet (18,34,50)?")

    # if mvit is True
    parser.add_argument("--src", type=str, required=False, default=None,
                        help="What is the variant of ResNet (18,34,50)?")

    # choose dataset
    parser.add_argument("--ten", type=bool, required=False, default=False, help="cifar10 will be used")
    parser.add_argument("--hundred", type=bool, required=False, default=False,
                        help="cifar100 will be used")

    args = parser.parse_args()

    dest = args.dest
    model = args.model
    gating_model = args.gating_model
    ucb = args.ucb

    resnet = args.resnet
    mvit = args.mvit

    variant = args.variant
    src = args.src

    ten = args.ten
    hundred = args.hundred

    final = Evaluate(dest=dest, model=model, gating_model=gating_model, ucb=ucb, hundred=hundred, ten=ten,
                     resnet=resnet, mvit=mvit, variant=variant, src=src)
    final.testing_data()
    final.evaluation()

if __name__ == "__main__":
    main()