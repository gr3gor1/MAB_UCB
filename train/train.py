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
        if self.ten:
            self.train_loader, self.val_loader = data_loader10(data_dir=self.dest + '/data',
                                                     batch_size=64)
        else:
            self.train_loader, self.val_loader = data_loader100(data_dir=self.dest + '/data',
                                                     batch_size=64)
        return

    def train(self):

        if self.resnet and self.base:
            train_resnet(train_loader=self.train_loader, dest=self.dest, num_epochs=self.num_epochs,
                         variant=self.variant)
        elif self.resnet and self.static:
            train_resnet_static(train_loader=self.train_loader, dest=self.dest, num_epochs=self.num_epochs,
                         variant=self.variant)
        elif self.resnet and self.dynamic:
            train_resnet_dynamic(dataloader=self.train_loader, dest=self.dest, epochs=self.num_epochs,
                                 variant=self.variant)
        elif self.mvit and self.base and self.ten:
            train_mvit(train_loader=self.train_loader, valid_loader=self.val_loader, dest=self.dest,
                       num_epochs=self.num_epochs, classes=10)
        elif self.mvit and self.static and self.ten:
            train_mvit_static(train_loader=self.train_loader, valid_loader=self.val_loader, dest=self.dest,
                              num_epochs=self.num_epochs, src=self.src, classes=10)
        elif self.mvit and self.dynamic and self.ten:
            train_mvit_dynamic(train_loader=self.train_loader, dest=self.dest, src=self.src, classes=10,
                               num_epochs=self.num_epochs)
        elif self.mvit and self.base and self.hundred:
            train_mvit(train_loader=self.train_loader, valid_loader=self.val_loader, dest=self.dest,
                       num_epochs=self.num_epochs, classes=100)
        elif self.mvit and self.static and self.hundred:
            train_mvit_static(train_loader=self.train_loader, valid_loader=self.val_loader, dest=self.dest,
                              num_epochs=self.num_epochs, src=self.src, classes=100)
        elif self.mvit and self.dynamic and self.hundred:
            train_mvit_dynamic(train_loader=self.train_loader, dest=self.dest, src=self.src, classes=100,
                               num_epochs=self.num_epochs)
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