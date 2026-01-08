# -*- coding: utf-8 -*-
import os
import torch
import numpy as np

from tqdm import tqdm
from itertools import repeat
from base import BaseTrainer
from utils import MetricTracker, Timer, model_complexity, calculate_aupr, calculate_auroc, plot_pr_curve, plot_roc_curve


class Trainer_Sequence(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, lr_scheduler, metric_ftns, config,
                 train_data_loader, valid_data_loader, test_data_loader):
        super().__init__(model, criterion, optimizer, lr_scheduler, metric_ftns, config)
        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader

        len_epoch = config['trainer'].get('len_epoch')
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = self._inf_loop(self.train_data_loader)
            self.len_epoch = len_epoch
        
        self.timer = Timer()
        self.log_step = int(np.sqrt(len(train_data_loader)))
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.timer.reset()

        # Training Part.
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.train_data_loader):
            input = {'TF': data['input']['TF_feature'].to(self.config.device),
                     'GENE': data['input']['GENE_feature'].to(self.config.device),
                     'TF_len': data['input']['TF_len'].to(self.config.device)}
            target = data['label'].to(self.config.device)
            batch_size = input['TF'].size(0)

            self.optimizer.zero_grad()
            output, _ = self.model(input)
            loss = self.criterion(output, target.to(torch.float32))
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item(), batch_size)
            output = output.detach().cpu().numpy() > 0.5
            target = target.detach().cpu().numpy()
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target), batch_size)

            # Iteration-Based Logging.
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'model')
            self.writer.add_scalar('loss_iteration', loss.item())
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))

            if batch_idx == self.len_epoch - 1:
                break

        # Validation Part.
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for _, data in enumerate(self.valid_data_loader):
                input = {'TF': data['input']['TF_feature'].to(self.config.device),
                        'GENE': data['input']['GENE_feature'].to(self.config.device),
                        'TF_len': data['input']['TF_len'].to(self.config.device)}
                target = data['label'].to(self.config.device)
                batch_size = input['TF'].size(0)

                output, _ = self.model(input)
                loss = self.criterion(output, target.to(torch.float32))

                self.valid_metrics.update('loss', loss.item(), batch_size)
                output = output.detach().cpu().numpy() > 0.5
                target = target.detach().cpu().numpy()
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target), batch_size)
        
        lr = self.lr_scheduler.get_last_lr()[0]
        duration = self.timer.check()
        self.lr_scheduler.step()

        train_log = self.train_metrics.result()
        self.writer.set_step(epoch, 'train')
        for key, value in train_log.items():
            self.writer.add_scalar(key, value)
        
        val_log = self.valid_metrics.result()
        self.writer.set_step(epoch, 'valid')
        for key, value in val_log.items():
            self.writer.add_scalar(key, value)
        
        self.writer.set_step(epoch, 'model')
        self.writer.add_scalar('lr', lr)
        self.writer.add_scalar('duration', duration)
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = {'lr': lr, 'duration': duration}
        log.update({'train_'+k : v for k, v in train_log.items()})
        log.update({'val_'+k : v for k, v in val_log.items()})
        return log        

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def _inf_loop(self, data_loader):
        ''' wrapper function for endless data loader. '''
        for loader in repeat(data_loader):
            yield from loader

    def test(self):
        """
        Test the model.
        :return: A log that contains average loss and metric.
        """
        model_path = os.path.join(self.config.model_dir, 'model_best.pth')
        self._resume_checkpoint(model_path)

        self.model.eval()
        output_list = []
        target_list = []
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.test_data_loader, ascii=True, disable=True)):
                input = {'TF': data['input']['TF_feature'].to(self.config.device),
                        'GENE': data['input']['GENE_feature'].to(self.config.device),
                        'TF_len': data['input']['TF_len'].to(self.config.device)}
                target = data['label'].to(self.config.device)
                output, _ = self.model(input)
                output_list.append(output)
                target_list.append(target)

        output = torch.cat(output_list, dim=0).detach().cpu().numpy()
        target = torch.cat(target_list, dim=0).detach().cpu().numpy()
        log = {met.__name__: met(output > 0.5, target) for _, met in enumerate(self.metric_ftns)}

        fpr_list, tpr_list, auroc = calculate_auroc(output, target)
        precision_list, recall_list, aupr = calculate_aupr(output, target)
        plot_roc_curve(fpr_list, tpr_list, os.path.join(self.config.result_dir, 'roc_curve.png'))
        plot_pr_curve(precision_list, recall_list, os.path.join(self.config.result_dir, 'pr_curve.png'))
        log.update({'auroc': auroc, 'aupr': aupr})

        if self.config['model']['args']['feature'] == 'onehot':
            input = {'TF': torch.randn(1, 1000, 20, dtype=torch.float32).to(self.config.device),
                     'GENE': torch.randn(1, 1000, 4, dtype=torch.float32).to(self.config.device),
                     'TF_len': torch.tensor([1000], dtype=torch.int64).to(self.config.device)}
        elif self.config['model']['args']['feature'] == 'vector':
            input = {'TF': torch.randn(1, 1000, 100, dtype=torch.float32).to(self.config.device),
                    'GENE': torch.randn(1, 1000, 100, dtype=torch.float32).to(self.config.device),
                    'TF_len': torch.tensor([1000], dtype=torch.int64).to(self.config.device)}
        elif self.config['model']['args']['feature'] == 'bert':
            input = {'TF': torch.randn(1, 1000, 26, dtype=torch.float32).to(self.config.device),
                    'GENE': torch.randn(1, 1000, 69, dtype=torch.float32).to(self.config.device),
                    'TF_len': torch.tensor([1000], dtype=torch.int64).to(self.config.device)}
        macs, params = model_complexity(self.model, [input])
        log.update({'macs': macs, 'params': params})
        self.logger.info(log)

        self.writer.set_step(0, 'test')
        self.writer.add_text('result', str(log))
