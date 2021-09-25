import torch.nn as nn
import torch


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.class_loss_func = ['CrossEntropy']
        self.loss = []
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Rank':
                loss_function = RankHingedLoss(margin=args.rank_margin)
            elif loss_type == 'Rela':
                loss_function = RelativeDistLoss()
            elif loss_type == 'CrossEntropy':
                loss_function = nn.CrossEntropyLoss()
            else:
                raise ValueError(f'Loss {loss_type} not supported !')

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

    def forward(self, input_mos, target_mos, input_class=None, target_class=None):
        loss_sum = 0.0
        loss_item = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] not in self.class_loss_func:
                    loss = l['function'](input_mos, target_mos)
                else:
                    loss = l['function'](input_class, target_class)
                effective_loss = l['weight'] * loss
                loss_item[l['type']] = loss
                loss_sum += effective_loss
        loss_item['Total'] = loss_sum
        return loss_sum, loss_item


class RankHingedLoss(torch.nn.Module):
    def __init__(self, margin=0.05, y_margin=0.01):
        super(RankHingedLoss, self).__init__()
        self.margin = margin
        self.y_margin = y_margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 2
        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0

    def forward(self, x_pred, y_true):
        self.check_type_forward((x_pred, y_true))
        bs = y_true.shape[0]

        X_pred = x_pred.repeat(bs, 1)
        X_diff = X_pred - X_pred.t()
        Y_true = y_true.repeat(bs, 1)
        Y_diff = Y_true - Y_true.t()

        Y_diff[torch.abs(Y_diff) < self.y_margin] = 0.0
        Y_diff_sign = torch.sign(Y_diff)

        rank_diff = torch.clamp(self.margin - Y_diff_sign * X_diff, min=0.0)
        rank_diff = torch.triu(rank_diff, diagonal=1)
        rank_loss = torch.sum(rank_diff) / (bs * (bs - 1) / 2)
        return rank_loss


class RelativeDistLoss(nn.Module):
    def __init__(self, margin=0.05):
        super(RelativeDistLoss, self).__init__()
        self.margin = margin

    def forward(self, pred, label):
        b = len(pred)

        pred_matrix = pred.repeat(pred.shape[0], 1)
        pred_matrix_2 = pred_matrix.t()

        label_matrix = label.repeat(label.shape[0], 1)
        label_matrix_2 = label_matrix.t()

        pred_rank = pred_matrix - pred_matrix_2
        label_rank = label_matrix - label_matrix_2

        loss = torch.sum(torch.abs(pred_rank - label_rank)) / (2 * b)
        return loss
