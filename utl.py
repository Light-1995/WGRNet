import os, torch, time
from utils.utils import save_config
from data.dataset import data
from data.data import get_data, get_data_val
from torch.utils.data import DataLoader

class BaseSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nEpochs = cfg['nEpochs']
        self.checkpoint_dir = cfg['checkpoint']
        self.epoch = 1

        timestamp = int(time.time())
        time_tuple = time.localtime(timestamp)
        self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time_tuple)



        if cfg['gpu_mode']:
            self.num_workers = cfg['threads']
        else:
            self.num_workers = 0

        self.train_dataset = get_data(cfg, cfg['data_dir_train'])
        self.train_loader = DataLoader(self.train_dataset, cfg['data']['batch_size'], shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True)#shuffle=True
        #(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,pin_memory=True, drop_last=True)
        self.val_dataset = get_data_val(cfg, cfg['data_dir_eval'])
        self.val_loader = DataLoader(self.val_dataset, cfg['data']['Val_batch_size'], shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True)
        #(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
        self.records = {'Epoch': [], 'Loss': [],'Val-Loss': [],'ERGAS': [], 'SAM': [], 'CC': [], 'SCC': [], 'PSNR': [], 'RMSE': []}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def load_checkpoint(self, model_path):
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            self.epoch = ckpt['epoch']
            self.records = ckpt['records']
        else:
            raise FileNotFoundError

    def save_checkpoint(self):
        self.ckp = {
            'epoch': self.epoch,
            'records': self.records,
        }

    def train(self):
        raise NotImplementedError
    
    def eval(self):
        raise NotImplementedError
    
    def run(self):
        while self.epoch <= self.nEpochs:
            self.train()
            self.eval()
            self.save_checkpoint()
            # self.save_records()
            self.epoch += 1
        #self.logger.log('Training done.')
