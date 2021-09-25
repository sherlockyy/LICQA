import time
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

from optimizer import make_optimizer
from utility import AverageMeter, IQAPerformance


class Trainer:
    def __init__(self, args, model, loader, loss):
        self.args = args
        self.log_dir = args.log_dir
        self.writer = SummaryWriter(self.log_dir)

        self.loader_train, self.loader_val, self.loader_test = loader.loader_train, loader.loader_val, loader.loader_test
        self.model = model
        self.loss = loss
        self.optimizer = make_optimizer(args, self.model)

        self.cur_epoch = 0
        self.cur_step = 0
        self.bestinfo = {'epoch': -1, 'loss': 1e6, 'srcc': 0.0, 'plcc': 0.0}
        self.epoch_step = int(len(self.loader_train))
        self.best_perf = -0.01

    def main_worker(self):        
        epoch_start = time.time()
        for epoch in range(1, self.args.epochs+1):
            self.cur_epoch = epoch            
            is_plot = epoch % self.args.save_scatter == 0

            train_loss, train_corr = self.train(is_plot=is_plot)
            self.optimizer.schedule()
            tr_time = time.time() - epoch_start

            self.writer.add_scalar('learning rate', self.optimizer.get_lr(), epoch)
            for k, v in train_loss.items():
                self.writer.add_scalars(k, {'train': v}, epoch)
            for k, v in train_corr.items():
                self.writer.add_scalars(k, {'train': v}, epoch)

            if epoch >= 100 and epoch % self.args.save_weights == 0:
                self.model.save(self.args.log_dir, f'checkoutpoint_ep{epoch:03d}.pth.tar')

            if epoch % self.args.test_every == 0:
                val_loss, val_corr = self.valid(is_plot=is_plot)
                test_loss, test_corr = self.test(is_plot=is_plot)

                is_best = val_corr['srcc'] > self.bestinfo['srcc']
                if is_best:
                    self.bestinfo['epoch'] = epoch
                    self.bestinfo['loss'] = val_loss['Total']
                    self.bestinfo['srcc'], self.bestinfo['plcc'] = val_corr['srcc'], val_corr['plcc']
                    self.model.save(self.args.log_dir, 'checkoutpoint_best.pth.tar')

                for k in val_loss.keys():
                    self.writer.add_scalars(k, {'val': val_loss[k]}, epoch)
                    self.writer.add_scalars(k, {'test': test_loss[k]}, epoch)
                for k in val_corr.keys():
                    self.writer.add_scalars(k, {'val': val_corr[k]}, epoch)
                    self.writer.add_scalars(k, {'test': test_corr[k]}, epoch)

            print(f"#Ep -> {epoch}/{self.args.epochs} | Time -> {tr_time:.1f}s | Best -> Ep {self.bestinfo['epoch']}, "
                  f"Loss {self.bestinfo['loss']:.4f}, SRCC {self.bestinfo['srcc']:.4f}")
        return self.bestinfo

    def train(self, is_plot=False):
        time_data = AverageMeter() 
        time_model = AverageMeter()

        train_loss = {}
        for l in self.loss.loss:
            train_loss[l['type']] = AverageMeter()
        train_perf = IQAPerformance(self.log_dir)

        self.model.train()
        time_start = time.time()
        for batch, (rx, dx, cy, sy) in enumerate(self.loader_train, start=1):
            rx, dx, cy, sy = self.prepare(rx, dx, cy, sy)

            time_mid = time.time()
            time_data.update(time_mid - time_start)

            cy_pred, sy_pred = self.model((rx, dx))
            self.optimizer.zero_grad()
            loss, loss_items = self.loss(input_mos=sy_pred, target_mos=sy, input_class=cy_pred, target_class=cy)

            for k, v in loss_items.items():
                train_loss[k].update(v.item(), rx.size(0))
            train_perf.update(sy_pred, sy)
            loss.backward()
            self.optimizer.step()

            time_model.update(time.time() - time_mid)
            time_start = time.time()

            self.cur_step = (self.cur_epoch - 1) * self.epoch_step + batch
            self.writer.add_scalar('batch_loss', loss.item(), self.cur_step)

        for k, l in train_loss.items():
            train_loss[k] = l.avg
        train_corr = train_perf.compute(is_plot=is_plot, fig_name=f'train_{self.cur_epoch}.png')

        print(f'Train Time -- Data {time_data.avg * 1000:.1f}ms | Model {time_model.avg * 1000:.1f}ms')
        print('      Loss -- ' + ' | '.join(f'{k} {v:.4f}' for k, v in train_loss.items()))
        print(f"      Corr -- SRCC {train_corr['srcc']:.4f} | PLCC {train_corr['plcc']:.4f} | RMSE {train_corr['rmse']:.4f}")
        return train_loss, train_corr

    def valid(self, is_plot=False):
        val_loss = {}
        for l in self.loss.loss:
            val_loss[l['type']] = AverageMeter()
        val_perf = IQAPerformance(self.log_dir)

        self.model.eval()
        for batch, (rx, dx, cy, sy) in enumerate(self.loader_val, start=1):
            rx, dx, cy, sy = self.prepare(rx, dx, cy, sy)
            cy_pred, sy_pred = self.model((rx, dx))
            loss, loss_items = self.loss(input_mos=sy_pred, target_mos=sy, input_class=cy_pred, target_class=cy)

            for k, v in loss_items.items():
                val_loss[k].update(v.item(), rx.size(0))
            val_perf.update(sy_pred, sy)

        for k, l in val_loss.items():
            val_loss[k] = l.avg
        val_corr = val_perf.compute(is_plot=is_plot, fig_name=f'val_{self.cur_epoch}.png')

        print('Val   Loss -- ' + ' | '.join(f'{k} {v:.4f}' for k, v in val_loss.items()))
        print(f"      Corr -- SRCC {val_corr['srcc']:.4f} | PLCC {val_corr['plcc']:.4f} | RMSE {val_corr['rmse']:.4f}")
        return val_loss, val_corr

    def test(self, is_plot=False):
        test_loss = {}
        for l in self.loss.loss:
            test_loss[l['type']] = AverageMeter()           
        test_perf = IQAPerformance(self.log_dir)

        self.model.eval()
        for batch, (rx, dx, cy, sy) in enumerate(self.loader_test, start=1):
            rx, dx, cy, sy = self.prepare(rx, dx, cy, sy)
            cy_pred, sy_pred = self.model((rx, dx))            
            loss, loss_items = self.loss(input_mos=sy_pred, target_mos=sy, input_class=cy_pred, target_class=cy)

            for k, v in loss_items.items():
                test_loss[k].update(v.item(), rx.size(0))
            test_perf.update(sy_pred, sy)
    
        for k, l in test_loss.items():
            test_loss[k] = l.avg
        test_corr = test_perf.compute(is_plot=is_plot, fig_name=f'test_{self.cur_epoch}.png')

        print('Test  Loss -- ' + ' | '.join(f'{k} {v:.4f}' for k, v in test_loss.items()))
        print(f"      Corr -- SRCC {test_corr['srcc']:.4f} | PLCC {test_corr['plcc']:.4f} | RMSE {test_corr['rmse']:.4f}")
        return test_loss, test_corr

    def prepare(self, *args):
        if self.args.device.type == 'cuda':
            return [a.cuda(non_blocking=True) for a in args]
        return args
