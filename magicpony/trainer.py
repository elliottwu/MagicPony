import os
import os.path as osp
import glob
from datetime import datetime
import torch
from .utils import meters, misc


def indefinite_generator(loader):
    while True:
        for x in loader:
            yield x


class Trainer:
    def __init__(self, cfgs, model):
        self.cfgs = cfgs
        self.device = cfgs.get('device', 'cpu')
        self.checkpoint_dir = cfgs.get('checkpoint_dir', 'results')
        self.save_checkpoint_freq = cfgs.get('save_checkpoint_freq', 1)
        self.keep_num_checkpoint = cfgs.get('keep_num_checkpoint', 2)  # -1 for keeping all checkpoints
        self.archive_code = cfgs.get('archive_code', True)
        self.resume = cfgs.get('resume', True)
        self.checkpoint_name = cfgs.get('checkpoint_name', None)
        self.test_result_dir = cfgs.get('test_result_dir', None)

        self.use_logger = cfgs.get('use_logger', True)
        self.log_image_freq = cfgs.get('log_image_freq', 1000)
        self.log_loss_freq = cfgs.get('log_loss_freq', 100)
        self.log_train = cfgs.get('log_train', False)
        self.log_val = cfgs.get('log_val', False)
        self.fix_log_batch = cfgs.get('fix_log_batch', False)
        self.save_train_result_freq = cfgs.get('save_train_result_freq', None)
        self.train_result_dir = osp.join(self.checkpoint_dir, 'training_results')

        self.num_epochs = cfgs.get('num_epochs', 1)
        self.batch_size = cfgs.get('batch_size', 64)
        self.in_image_size = cfgs.get('in_image_size', 256)
        self.out_image_size = cfgs.get('out_image_size', 256)
        self.data_type = cfgs.get('data_type', 'sequence')

        self.train_data_dir = cfgs.get('train_data_dir', None)
        self.val_data_dir = cfgs.get('val_data_dir', None)
        self.test_data_dir = cfgs.get('test_data_dir', None)
        self.num_workers = cfgs.get('num_workers', 4)
        self.train_loader, self.val_loader, self.test_loader = model.get_data_loaders(cfgs, self.data_type, in_image_size=self.in_image_size, out_image_size=self.out_image_size, batch_size=self.batch_size, num_workers=self.num_workers, train_data_dir=self.train_data_dir, val_data_dir=self.val_data_dir, test_data_dir=self.test_data_dir)

        self.current_epoch = 0
        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.model = model(cfgs)
        self.model.trainer = self

    def load_checkpoint(self, load_optim=True):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if self.checkpoint_name is not None:
            checkpoint_path = osp.join(self.checkpoint_dir, self.checkpoint_name)
        else:
            checkpoints = sorted(glob.glob(osp.join(self.checkpoint_dir, '*.pth')))
            if len(checkpoints) == 0:
                return 0, 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = osp.basename(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_model_state(cp)
        if load_optim:
            self.model.load_optimizer_state(cp)
        self.metrics_trace = cp.get('metrics_trace', self.metrics_trace)
        epoch = cp.get('epoch', 999)
        total_iter = cp.get('total_iter', 999999)
        return epoch, total_iter

    def save_checkpoint(self, epoch, total_iter=0, save_optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified epoch."""
        misc.xmkdir(self.checkpoint_dir)
        checkpoint_path = osp.join(self.checkpoint_dir, f'checkpoint{epoch:04}.pth')
        state_dict = self.model.get_model_state()
        if save_optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        state_dict['total_iter'] = total_iter
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            misc.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        assert self.test_loader is not None, "test_data_dir must be specified for testing"
        self.model.to(self.device)
        self.model.set_eval()
        epoch, self.total_iter = self.load_checkpoint(load_optim=False)

        if self.test_result_dir is None:
            self.test_result_dir = osp.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth', ''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            for iteration, batch in enumerate(self.test_loader):
                m = self.model.forward(batch, epoch=epoch, total_iter=self.total_iter, save_results=True, save_dir=self.test_result_dir, is_training=False)
                print(f"T{epoch:04}/{iteration:05}")

    def train(self):
        """Perform training."""
        assert self.train_loader is not None, "train_data_dir must be specified for training"

        # archive code and configs
        if self.archive_code:
            misc.archive_code(osp.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py'])
        misc.dump_yaml(osp.join(self.checkpoint_dir, 'configs.yml'), self.cfgs)

        # initialize
        start_epoch = 0
        self.total_iter = 0
        self.metrics_trace.reset()
        self.model.to(self.device)
        self.model.reset_optimizers()
        self.model.set_train()

        # resume from checkpoint
        if self.resume:
            start_epoch, self.total_iter = self.load_checkpoint(load_optim=True)

        # initialize tensorboard logger
        if self.use_logger:
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(osp.join(self.checkpoint_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")), flush_secs=10)
            if self.log_val:
                assert self.val_loader is not None, "val_data_dir must be specified for logging validation"
                self.val_data_iterator = indefinite_generator(self.val_loader)
            if self.fix_log_batch:
                self.log_batch = next(self.val_data_iterator)

        # run epochs
        epoch = 0
        for epoch in range(start_epoch, self.num_epochs):
            metrics = self.run_train_epoch(epoch)
            self.metrics_trace.append("train", metrics)
            if (epoch+1) % self.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch+1, total_iter=self.total_iter, save_optim=True)
            self.metrics_trace.save(osp.join(self.checkpoint_dir, 'metrics.json'))
        print(f"Training completed for all {epoch+1} epochs.")

    def run_train_epoch(self, epoch):
        metrics = self.make_metrics()
        for iteration, batch in enumerate(self.train_loader):
            self.total_iter += 1

            m = self.model.forward(batch, epoch=epoch, total_iter=self.total_iter, is_training=True)
            self.model.backward()

            num_seqs, num_frames = batch[0].shape[:2]
            total_im_num = num_seqs*num_frames
            metrics.update(m, total_im_num)
            print(f"T{epoch:04}/{iteration:05}/{metrics}")

            if self.use_logger:
                if self.total_iter % self.log_loss_freq == 0:
                    for name, loss in m.items():
                        self.logger.add_scalar(f'train_loss/{name}', loss, self.total_iter)

                if self.save_train_result_freq is not None and self.total_iter % self.save_train_result_freq == 0:
                    with torch.no_grad():
                        m = self.model.forward(batch, epoch=epoch, total_iter=self.total_iter, save_results=True, save_dir=self.train_result_dir, is_training=False)
                        torch.cuda.empty_cache()

                if self.total_iter % self.log_image_freq == 0:
                    with torch.no_grad():
                        if self.log_train:
                            m = self.model.forward(batch, epoch=epoch, logger=self.logger, total_iter=self.total_iter, logger_prefix='train_', is_training=True)

                        if self.log_val:
                            if self.fix_log_batch:
                                batch = self.log_batch
                            else:
                                batch = next(self.val_data_iterator)
                            m = self.model.forward(batch, epoch=epoch, logger=self.logger, total_iter=self.total_iter, logger_prefix='val_', is_training=False)
                            for name, loss in m.items():
                                self.logger.add_scalar(f'val_loss/{name}', loss, self.total_iter)
                    torch.cuda.empty_cache()

        self.model.scheduler_step()
        return metrics
