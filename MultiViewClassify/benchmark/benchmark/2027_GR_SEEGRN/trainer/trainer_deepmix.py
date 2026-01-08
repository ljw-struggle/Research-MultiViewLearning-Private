# -*- coding: utf-8 -*-
import os
import torch
import numpy as np

from tqdm import tqdm
from itertools import repeat
from base import BaseTrainer
from utils import MetricTracker, Timer, model_complexity, calculate_aupr, calculate_auroc, plot_pr_curve, plot_roc_curve, calculate_ep


class Trainer_Mix(BaseTrainer):
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

        self.gene_node_expression = self.train_data_loader.get_node_expression().to(self.config.device)
        self.prior_graph_topology = self.train_data_loader.get_prior_graph_topology().to(self.config.device)
        self.index_gene_dict = self.train_data_loader.get_index_gene_dict()
        self.gene_index_dict = self.train_data_loader.get_gene_index_dict()

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
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

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
            graph_input = {'NODE': self.gene_node_expression,
                           'ENCODE': self.prior_graph_topology,
                           'DECODE': data['input']['EDGE'].transpose(0, 1).to(self.config.device)}
            branch_input = {'TF': data['input']['TF_feature'].to(self.config.device),
                            'TF_len': data['input']['TF_len'].to(self.config.device),
                            'GENE': data['input']['GENE_feature'].to(self.config.device)}

            target = data['label'].to(self.config.device)
            weight = data['weight'].to(self.config.device)
            batch_size = branch_input['TF'].size(0)

            self.optimizer.zero_grad()
            output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = self.model(graph_input, branch_input)
            if self.model.fusion_method == 'mmdynamics':
                loss = self.criterion(output, target.to(torch.long)) + 0.1 * self.model.mmdynamics_loss(classifier_expression, confidence_expression, classifier_sequence, confidence_sequence, target.to(torch.long), weight)
            else:
                loss = self.criterion(output, target.to(torch.long))
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update('loss', loss.item(), batch_size)
            output = torch.softmax(output, dim=1)[:, 1]
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output > 0.5, target), batch_size)

            # Iteration-Based Logging.
            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx, 'model')
            self.writer.add_scalar('loss_iteration', loss.item())
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), loss.item()))

            if batch_idx == self.len_epoch - 1:
                break
        
        train_log = self.train_metrics.result()

        # Validation Part.
        self.model.eval()
        self.valid_metrics.reset()
        output_list = []
        target_list = []
        with torch.no_grad():
            for _, data in enumerate(self.valid_data_loader):
                graph_input = {'NODE': self.gene_node_expression,
                               'ENCODE': self.prior_graph_topology,
                               'DECODE': data['input']['EDGE'].transpose(0, 1).to(self.config.device)}
                branch_input = {'TF': data['input']['TF_feature'].to(self.config.device),
                                'TF_len': data['input']['TF_len'].to(self.config.device),
                                'GENE': data['input']['GENE_feature'].to(self.config.device)}
                
                target = data['label'].to(self.config.device)
                batch_size = branch_input['TF'].size(0)

                output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = self.model(graph_input, branch_input)
                loss = self.criterion(output, target.to(torch.long))

                self.valid_metrics.update('loss', loss.item(), batch_size)
                output = torch.softmax(output, dim=1)[:, 1]
                output_list.append(output)
                target_list.append(target)

            output = torch.cat(output_list, dim=0).detach().cpu().numpy()
            target = torch.cat(target_list, dim=0).detach().cpu().numpy()
            batch_size = output.shape[0]
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(output > 0.5, target), batch_size)

            _, _, auroc = calculate_auroc(output, target)
            _, _, aupr = calculate_aupr(output, target)
            ep, epr = calculate_ep(output, target)
            valid_metrics = {'auroc': auroc, 'aupr': aupr, 'ep': ep, 'epr': epr}

        val_log = self.valid_metrics.result()
        val_log.update(valid_metrics)

        # Test Part.
        self.model.eval()
        self.test_metrics.reset()
        output_list = []
        target_list = []
        with torch.no_grad():
            for _, data in enumerate(self.test_data_loader):
                graph_input = {'NODE': self.gene_node_expression,
                            'ENCODE': self.prior_graph_topology,
                            'DECODE': data['input']['EDGE'].transpose(0, 1).to(self.config.device)}
                branch_input = {'TF': data['input']['TF_feature'].to(self.config.device),
                                'TF_len': data['input']['TF_len'].to(self.config.device),
                                'GENE': data['input']['GENE_feature'].to(self.config.device)}

                target = data['label'].to(self.config.device)
                batch_size = branch_input['TF'].size(0)
                
                output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = self.model(graph_input, branch_input)
                loss = self.criterion(output, target.to(torch.long))

                self.test_metrics.update('loss', loss.item(), batch_size)
                output = torch.softmax(output, dim=1)[:, 1]
                output_list.append(output)
                target_list.append(target)

            output = torch.cat(output_list, dim=0).detach().cpu().numpy()
            target = torch.cat(target_list, dim=0).detach().cpu().numpy()
            batch_size = output.shape[0]
            for met in self.metric_ftns:
                self.test_metrics.update(met.__name__, met(output > 0.5, target), batch_size)
            
            _, _, auroc = calculate_auroc(output, target)
            _, _, aupr = calculate_aupr(output, target)
            ep, epr = calculate_ep(output, target)
            test_metrics = {'auroc': auroc, 'aupr': aupr, 'ep': ep, 'epr': epr}

        test_log = self.test_metrics.result()
        test_log.update(test_metrics)

        lr = self.lr_scheduler.get_last_lr()[0]
        duration = self.timer.check()
        self.lr_scheduler.step()

        self.writer.set_step(epoch, 'train')
        for key, value in train_log.items():
            self.writer.add_scalar(key, value)
        
        self.writer.set_step(epoch, 'valid')
        for key, value in val_log.items():
            self.writer.add_scalar(key, value)

        self.writer.set_step(epoch, 'test')
        for key, value in test_log.items():
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
        log.update({'test_'+k : v for k, v in test_log.items()})
        return log        

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.train_data_loader)
        return base.format(current, total, 100.0 * current / total)
    
    def _inf_loop(data_loader):
        ''' wrapper function for endless data loader. '''
        for loader in repeat(data_loader):
            yield from loader

    def test(self):
        """
        Test the model.
        :return: A log that contains average loss and metric.
        """
        if self.config.resume is not None:
            self._resume_checkpoint(self.config.resume)

        self.model.eval()
        self.test_metrics.reset()
        output_list = []
        target_list = []
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.test_data_loader, ascii=True, disable=True)):
                graph_input = {'NODE': self.gene_node_expression,
                               'ENCODE': self.prior_graph_topology,
                               'DECODE': data['input']['EDGE'].transpose(0, 1).to(self.config.device)}
                branch_input = {'TF': data['input']['TF_feature'].to(self.config.device),
                                'TF_len': data['input']['TF_len'].to(self.config.device),
                                'GENE': data['input']['GENE_feature'].to(self.config.device)}

                target = data['label'].to(self.config.device)
                output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = self.model(graph_input, branch_input)
                output = torch.softmax(output, dim=1)[:, 1]                 
                output_list.append(output)
                target_list.append(target)

        output = torch.cat(output_list, dim=0).detach().cpu().numpy()
        target = torch.cat(target_list, dim=0).detach().cpu().numpy()
        log = {met.__name__: met(output > 0.5, target) for _, met in enumerate(self.metric_ftns)}

        fpr_list, tpr_list, auroc = calculate_auroc(output, target)
        precision_list, recall_list, aupr = calculate_aupr(output, target)
        aupr_norm = aupr / np.mean(target)
        plot_roc_curve(fpr_list, tpr_list, os.path.join(self.config.result_dir, 'roc_curve.png'))
        plot_pr_curve(precision_list, recall_list, os.path.join(self.config.result_dir, 'pr_curve.png'))
        ep, epr = calculate_ep(output, target)
        log.update({'auroc': auroc, 'aupr': aupr, 'aupr_norm': aupr_norm, 'ep': ep, 'epr': epr})

        if self.config['loader']['args']['feature'] == 'onehot':
            protein_sequence_feature_dim = 20
            gene_sequence_feature_dim = 4
        if self.config['loader']['args']['feature'] == 'vector':
            protein_sequence_feature_dim = 100
            gene_sequence_feature_dim = 100
        if self.config['loader']['args']['feature'] == 'bert':
            protein_sequence_feature_dim = 26
            gene_sequence_feature_dim = 69
        graph_input = {'NODE': self.gene_node_expression,
                        'ENCODE': self.prior_graph_topology,
                        'DECODE': torch.tensor([[1, 1]], dtype=torch.int64).transpose(0, 1).to(self.config.device)}
        branch_input = {'TF': torch.randn(1, 1000, protein_sequence_feature_dim, dtype=torch.float32).to(self.config.device),
                        'TF_len': torch.tensor([1000], dtype=torch.int64).to(self.config.device),
                        'GENE': torch.randn(1, 1000, gene_sequence_feature_dim, dtype=torch.float32).to(self.config.device)}
        macs, params = model_complexity(self.model, [graph_input, branch_input])
        memory = str(round(torch.cuda.memory_allocated(self.config.device)/1024/1024/1024, 3)) + 'G'
        log.update({'macs': macs, 'params': params, 'memory': memory})

        self.logger.info(log)
        self.writer.set_step(0, 'test')
        self.writer.add_text('result', str(log))
