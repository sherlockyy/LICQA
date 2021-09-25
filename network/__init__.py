from pathlib import Path
from importlib import import_module

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')

        self.device = args.device
        module = import_module('network.' + args.model.lower())
        self.model = module.make_model(args)
        self.load(pre_train=args.pre_train)
        if self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(self.model).cuda()
        print(self.model)

    def load(self, pre_train):
        if pre_train:
            load_from = torch.load(pre_train, map_location=lambda storage, loc: storage)
            # load_from_new = {}
            # for k, v in load_from.items():
            #     if k.startswith('module.'):
            #         load_from_new[k[7:]] = v
            self.model.load_state_dict(load_from, strict=True)

    def save(self, apath, filename):
        save_dir = Path(apath) / 'checkoutpoints'
        save_dir.mkdir(exist_ok=True)
        save_file = save_dir / filename
        torch.save(self.model.module.state_dict(), save_file)

    def forward(self, data):
        return self.model.forward(data)
