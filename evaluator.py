from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

import network
from dataset import get_transform, IQADatabase, ValDatabase


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.save_path = args.predict_res
        self.transform = get_transform(args, phase='test')
        self.models = []

        model_checkpoints = args.pre_train.split('+')
        for checkpoint in model_checkpoints:
            args.pre_train = checkpoint
            self.models.append(network.Model(args))

        if self.args.test_on_fold is None:
            ref_dst = self.get_val_set_images()
            self.dataset = ValDatabase(args.data_dir, ref_dst, self.transform)
            self.test_on = 'val'
        else:
            test_files = [Path(self.args.train_files) / f'split{i}.csv' for i in range(1, 6) if i != args.test_on_fold]
            test_files.append(Path(self.args.train_files) / f'split{args.test_on_fold}.csv')
            self.dataset = IQADatabase(args.data_dir, test_files, self.transform, args.mos_norm)
            self.test_on = 'train'

        self.dst_imgs = self.dataset.dst_imgs
        self.img_num = len(self.dst_imgs)
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=not args.device.type == 'cpu',
            num_workers=args.n_threads,
        )

    def predict(self):
        mos_array = None
        pred_array = np.zeros((self.img_num, len(self.models)))

        for i, model in enumerate(self.models):
            if self.test_on == 'val':
                pred, mos = self.pred_val_set(model)
            elif self.test_on == 'train':
                pred, mos = self.pred_train_set(model)
            pred_array[:, i] = pred

        if mos:
            mos_array = np.array(mos)
        pred_mean = np.mean(pred_array, axis=1, keepdims=True)

        with open(self.save_path, 'w') as f:
            if mos_array is not None:
                for i in range(self.img_num):
                    f.write(f"{self.dst_imgs[i].name},{100 * float(pred_mean[i]):.3f},{100 * mos_array[i]:.3f}\n")
            else:
                for i in range(self.img_num):
                    f.write(f"{self.dst_imgs[i].name},{100 * float(pred_mean[i]):.3f}\n")

    def pred_train_set(self, model):
        model.eval()
        pred_list = []
        mos_list = []
        for i, (rx, dx, cy, sy) in enumerate(self.loader, start=1):
            _, y_pred = model((rx, dx))
            pred_list.extend([p.item() for p in y_pred])
            mos_list.extend([s.item() for s in sy])
        return pred_list, mos_list

    def pred_val_set(self, model):
        model.eval()
        pred_list = []
        for i, (rx, dx) in enumerate(self.loader, start=1):
            _, y_pred = model((rx, dx))
            pred_list.extend([p.item() for p in y_pred])
        return pred_list, None

    def get_val_set_images(self):
        ref_root = Path(self.args.data_dir) / 'Valid_Ref'
        dst_root = Path(self.args.data_dir) / 'Valid_Dis'

        dst_images = list(dst_root.glob('*.bmp'))
        dst_images.sort()
        ref_images = []
        dst_num = len(dst_images)

        for dst in dst_images:
            dst_name = dst.stem
            ref_name = dst_name.split('_')[0] + '.bmp'
            ref_images.append(ref_root / ref_name)

        ref_num = len(list(set(ref_images)))
        print(f"Predict score for {ref_num} reference images -> {dst_num} distortion images")
        return ref_images, dst_images
