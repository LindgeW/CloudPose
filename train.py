import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'mydataset'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# from pt_utils import BNMomentumScheduler
from CloudPose import CloudPose_all
from losses import get_loss
from dataset import MyDataset
from eval import trans_err_eval, pose_add
from scipy.spatial.transform import Rotation as Rot
from plyfile import PlyData

def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn


class BNMomentumScheduler(object):
    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mydataset', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=256, help='Point Number [256/512/1024] [default: 256]')
parser.add_argument('--num_class', type=int, default=1, help='class Number [default: 5]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 90]')
parser.add_argument('--optimizer', default='adam', help='adam or gd [default: adam]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=8e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=40, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='35,55,70',
                    help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
args = parser.parse_args()


BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
MAX_EPOCH = args.max_epoch
BASE_LEARNING_RATE = args.learning_rate
BN_DECAY_STEP = args.bn_decay_step
BN_DECAY_RATE = args.bn_decay_rate
NUM_CLASS = args.num_class
LR_DECAY_STEPS = [int(x) for x in args.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in args.lr_decay_rates.split(',')]
assert (len(LR_DECAY_STEPS) == len(LR_DECAY_RATES))
LOG_DIR = args.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = args.checkpoint_path if args.checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and args.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)' % LOG_DIR)
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(args) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # np.random.seed(np.random.get_state()[1][0] + worker_id)


sys.path.append(os.path.join(ROOT_DIR, args.dataset))

train_dataset = MyDataset('train', num_points=NUM_POINT)
test_dataset = MyDataset('val', num_points=NUM_POINT)
print(len(train_dataset), len(test_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=1, worker_init_fn=my_worker_init_fn)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=1, worker_init_fn=my_worker_init_fn)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = CloudPose_all(3)

# if torch.cuda.device_count() > 1:
#     log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     net = nn.DataParallel(net)
net.to(device)

criterion = get_loss

optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=args.weight_decay)

# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.9
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE ** (int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)


def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate_one_epoch():
    stat_dict = {}  # collect statistics
    net.eval()  # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(test_dataloader):
        if batch_idx % 10 == 0:
            print('Eval batch: %d' % batch_idx)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        loss, end_points = criterion(end_points)
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))))

    mean_loss = stat_dict['total_loss'] / float(batch_idx + 1)
    return mean_loss


def inference():
    def load_ply_vtx(pth):
        """
        load object vertices
        :param pth: str
        :return: pts: (N, 3)
        """
        ply = PlyData.read(pth)
        vtx = ply['vertex']
        pts = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)  # 一个顶点一行，每个顶点包含浮点xyz坐标值
        return pts

    print('inference on test set ...')
    model_points = load_ply_vtx('scans/ape.ply')
    net.eval()
    all_pose_pred = []
    all_pose_tgt = []
    for batch_idx, batch_data_label in enumerate(test_dataloader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        pred_rot = Rot.from_rotvec(end_points['axag_pred'].cpu().numpy()).as_matrix()
        pred_trans = end_points['translate_pred'].unsqueeze(-1).cpu().numpy()
        poses_pred = np.concatenate((pred_rot, pred_trans), axis=-1)
        poses_tgt = end_points['rot_trans_label'].cpu().numpy()
        all_pose_pred.append(poses_pred)
        all_pose_tgt.append(poses_tgt)

    poses_pred = np.concatenate(all_pose_pred, axis=0)
    poses_tgt = np.concatenate(all_pose_tgt, axis=0)
    ape_add = pose_add(poses_pred, poses_tgt, model_points)
    cmd_deg = trans_err_eval(poses_pred, poses_tgt)[1]
    print('ADD:', ape_add)
    print('x-cm x-deg:', cmd_deg)


def train_one_epoch(epoch):
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    net.train()   # set model to training mode
    for batch_idx, batch_data_label in enumerate(train_dataloader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 2
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


def set_seeds(seed=1347):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(start_epoch):
    global EPOCH_CNT
    min_loss = 1e10
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f' % (bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        set_seeds()
        train_one_epoch(epoch)
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9:  # Eval every 10 epochs
            loss = evaluate_one_epoch()
            inference()
            if loss < min_loss:
                min_loss = loss
                save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                             'optimizer_state_dict': optimizer.state_dict(),
                             'loss': min_loss}
                try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
                    save_dict['model_state_dict'] = net.module.state_dict()
                except:
                    save_dict['model_state_dict'] = net.state_dict()
                torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
    inference()


if __name__ == '__main__':
    train(start_epoch)
