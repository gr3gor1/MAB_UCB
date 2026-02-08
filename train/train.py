from mobilevit_base import *
from mobilevit_ee_static import *
from mobilevit_ee_dynamic import *

from resnet_base import *
from resnet_ee_static  import *
from resnet_ee_dynamic import *

from data.dataloader10 import *
from data.dataloader100 import *

import argparse

class Train:

    def __init__(self, dest, num_epochs, resnet, mvit, variant, src, base, static, dynamic, ten, hundred):

        if (resnet is False) and (mvit is False):
            raise ValueError("type of architecture needs to be selected")
        if (static is False) and (dynamic is False) and (base is False):
            raise ValueError("type of training needs to be selected")
        if (ten is False) and (hundred is False):
            raise ValueError("type of dataset needs to be selected")
        if resnet and (variant is None):
            raise ValueError("type of resnet variant needs to be selected")
        if dest is None:
            raise ValueError("destination path needs to be selected")
        if mvit and (src is None):
            raise ValueError("source path of backbone needs to be selected")

        self.dest = dest
        self.num_epochs = num_epochs
        self.resnet = resnet
        self.mvit = mvit
        self.variant = variant
        self.src = src
        self.base = base
        self.static = static
        self.dynamic = dynamic
        self.ten = ten
        self.hundred = hundred

        self.train_loader = None
        self.val_loader = None

        return

    def load_data(self):
        return

    def train(self):
        return

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dest", type=str, required=True, default=None, help="Where to store the models?")
    parser.add_argument("--num_epochs", type=int, required=True, default=20, help="How many epochs?")

    parser.add_argument("--resnet", type=bool, required=False, default=False,
                        help="train a resnet variant")
    parser.add_argument("--mvit", type=bool, required=False, default=False, help="train a mvit variant")

    # if resnet is True
    parser.add_argument("--variant", type=int, required=False, default=None,
                        help="What is the variant of ResNet (18,34,50)?")

    # if mvit is True
    parser.add_argument("--src", type=str, required=False, default=None,
                        help="What is the variant of ResNet (18,34,50)?")

    # choose mode
    parser.add_argument("--base", type=bool, required=False, default=False, help="non-adaptive training")
    parser.add_argument("--static", type=bool, required=False, default=False, help="static training")
    parser.add_argument("--dynamic", type=bool, required=False, default=False, help="dynamic training")

    # choose dataset
    parser.add_argument("--ten", type=bool, required=False, default=False, help="cifar10 will be used")
    parser.add_argument("--hundred", type=bool, required=False, default=False,
                        help="cifar100 will be used")


    args = parser.parse_args()

    dest = args.dest
    num_epochs = args.num_epochs

    resnet = args.resnet
    mvit = args.mvit

    variant = args.variant
    src = args.src

    base = args.base
    static = args.static
    dynamic = args.dynamic

    ten = args.ten
    hundred = args.hundred

    final = Train(dest=dest, num_epochs=num_epochs, resnet=resnet, mvit=mvit, variant=variant, src=src, base=base,
                  static=static, dynamic=dynamic, ten=ten, hundred=hundred)
    final.load_data()
    final.train()

if __name__ == "__main__":
    main()