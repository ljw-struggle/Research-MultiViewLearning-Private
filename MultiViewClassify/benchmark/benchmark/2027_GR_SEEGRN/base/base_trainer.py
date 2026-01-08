# -*- coding: utf-8 -*-
import os
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, optimizer, lr_scheduler, metric_ftns, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metric_ftns = metric_ftns
        self.init_epoch = 0

        self.epochs = config['trainer']['epochs']
        self.monitor = config['trainer'].get('monitor', 'off')
        self.early_stop = config['trainer'].get('early_stop', inf)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf

        self.result_dir = config.result_dir
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(self.log_dir, self.logger, config['trainer']['tensorboard'])

    @abstractmethod
    def _train_epoch(self, epoch, device):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        if self.config.resume is not None:
            self._resume_checkpoint(self.config.resume)

        not_improved_count = 0
        for epoch in range(self.init_epoch + 1, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    self.config.resume = self._save_checkpoint(epoch, save_best=True)
                else:
                    not_improved_count += 1
                    self._save_checkpoint(epoch, save_best=False)

                if not_improved_count == self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if self.mnt_mode == 'off':
                self.config_resume = self._save_checkpoint(epoch, save_best=True)
                self._save_checkpoint(epoch, save_best=False)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        state = {
            'epoch': epoch,
            'model': type(self.model).__name__,
            'criterion': type(self.criterion).__name__,
            'optimizer': type(self.optimizer).__name__,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        if save_best:
            best_filename = os.path.join(self.model_dir, 'model_best.pth')
            torch.save(state, best_filename)
            self.logger.info("Saving current best: model_best.pth ...")
            return best_filename
        else:
            filename = os.path.join(self.model_dir, 'checkpoint-epoch-{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
            return filename

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        self.init_epoch = checkpoint['epoch']

        # load model state from checkpoint.
        if checkpoint['model'] != self.config['model']['type']:
            self.logger.warning("Warning: model type given in config file is different from that of checkpoint.")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # load optimizer state from checkpoint.
        if checkpoint['optimizer'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.logger.info("Checkpoint loaded ... ")
