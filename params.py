import argparse


parser = argparse.ArgumentParser(description='Deep Image Quality Assessment for Learning-based Codecs')

# Hardware specifications
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--n_threads', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')

# Data specifications
parser.add_argument('--data_dir', type=str, default='../database', help='dataset directory')
parser.add_argument('--train_files', type=str, default='./train_files', help='train file path')
parser.add_argument('--resize', type=int, default=None, help='resize input image')
parser.add_argument('--crop_train', type=int, default=None, help='crop size for train set')
parser.add_argument('--crop_test', type=int, default=None, help='crop size for test set')
parser.add_argument('--hori_flip', type=float, default=None, help='probability of random horizontal flip')
parser.add_argument('--multi_crop_size', type=int, default=None,
                    help='if provided, image will be cropped to multi patches without overlap for training')
parser.add_argument('--multi_crop_num', type=int, default=1,
                    help='if {multi_crop_size} provided, image will be cropped to {multi_crop_num} patches for training')
parser.add_argument('--mean', type=str, default='(0.485, 0.456, 0.406)', help='normalize mean')
parser.add_argument('--std', type=str, default='(0.229, 0.224, 0.225)', help='normalize std')
parser.add_argument('--mos_norm', type=str, default='(0.00, 100.00)', help='mos normalization, min and max values')

# Model specifications
parser.add_argument('--model', default='vit', help='model arch name')

# Option for Residual network (ResNet)
parser.add_argument('--block', type=str, default='basicblock', choices=('basicblock', 'bottleneck'),
                    help='type of residual block')
parser.add_argument('--layers', type=str, default='[2, 2, 2, 2]', help='layer settings of ResNet')

# Option for (Wa)DIQaM-FR Model
parser.add_argument('--weighted_average', type=bool, default=True, help='weighted average or not')

# Option for DISTS Model
parser.add_argument('--vgg16_pretrain', type=str, default='../PTH/vgg16-397923af.pth',
                    help='weight path of prerained backbone network')
parser.add_argument('--add_l2pooling', action='store_true', help='whether add L2pooling after each stage')

# Option for ViT Model
parser.add_argument('--vit_input_size', type=int, default=256, help='input image size for ViT')
parser.add_argument('--linear_dim', type=int, default=512,
                    help='last dimension of output tensor after linear transformation')
parser.add_argument('--ts_blk', type=int, default=6, help='number of transformer blocks')
parser.add_argument('--att_head', type=int, default=16, help='number of heads in multi-head attention layer')
parser.add_argument('--mlp_dim', type=int, default=128, help='dimension of the MLP (FeedForward) layer')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--emb_dropout', type=float, default=0.1, help='embedding dropout rate')
parser.add_argument('--backbone', type=str, default='resnet18', help='type of backbone network')
parser.add_argument('--vit_pretrain', type=str, default='../PTH/resnet18-5c106cde.pth',
                    help='weight path of prerained backbone network')
parser.add_argument('--position_embedding', type=str, default='learned', choices=('sine', 'learned'),
                    help='position_embedding type')
parser.add_argument('--use_layer', type=str, default="1-4", help='get feature map from layer[start] to layer[end]')

# Training specifications
parser.add_argument('--exp_id', type=int, default=1, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                    help='exp id, exp{i}/train.csv, val.csv and test.csv are used for split')
parser.add_argument('--test_every', type=int, default=1, help='do test per every N epochs')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')

# Testing specifications
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--pre_train', type=str, default=None, help='pre-trained model path')
parser.add_argument('--predict_res', type=str, default=None, help='where to save predicted results')

# Optimization specifications
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--decay', type=str, default='20-40-60-80', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.8, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMW'),
                    help='optimizer to use (SGD | ADAM | RMSprop | ADAMW)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1+0.01*Rela+0.1*CrossEntropy',
                    help='loss function weights and types')
parser.add_argument('--rank_margin', type=float, default=0.05, help='margin param of rank loss function')

# Log specifications
parser.add_argument('--log_root', type=str, default='./log', help='directory for saving model weights and log file')
parser.add_argument('--save_weights', type=int, default=20, help='how many epochs to wait before saving model weights')
parser.add_argument('--save_scatter', type=int, default=20, help='how many epochs to wait before saving scatter plot')

args = parser.parse_args()

args.mean = tuple(map(float, args.mean.strip('()').split(',')))
args.std = tuple(map(float, args.std.strip('()').split(',')))
args.mos_norm = tuple(map(float, args.mos_norm.strip('()').split(',')))

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
