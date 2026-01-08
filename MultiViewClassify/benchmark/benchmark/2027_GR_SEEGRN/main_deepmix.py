# -*- coding: utf-8 -*-
import os
import time
import torch
import argparse
import collections

from loader import loader_deepmix as module_loader
from model import model as module_model
from model import criterion as module_criterion
from model import metric as module_metric

from tqdm import tqdm
from trainer import Trainer_Mix as Trainer
from parse_config import ConfigParser
from utils import model_complexity, calculate_aupr, calculate_auroc, plot_pr_curve, plot_roc_curve, calculate_ep


def train(config):
    logger = config.get_logger('train')

    train_data_loader = config.init_obj('loader', module_loader, logger=logger, mode='train')
    valid_data_loader = config.init_obj('loader', module_loader, logger=logger, mode='valid')
    test_data_loader = config.init_obj('loader', module_loader, logger=logger, mode='test')

    model = config.init_obj('model', module_model)
    logger.info(model)

    device, device_ids = config.device, config.device_ids
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = config.init_obj('criterion', module_criterion, device=device)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer_specifc = Trainer(model, criterion, optimizer, lr_scheduler, metrics, config=config,
                              train_data_loader=train_data_loader, valid_data_loader=valid_data_loader, test_data_loader=test_data_loader)
    
    trainer_specifc.train()
    trainer_specifc.test()


def test(config):
    logger = config.get_logger('test')

    test_data_loader = config.init_obj('loader', module_loader, logger=logger, mode='test')

    model = config.init_obj('model', module_model)
    logger.info(model)

    metric_ftns = [getattr(module_metric, met) for met in config['metrics']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    if config.resume is not None:
        model.load_state_dict(torch.load(config.resume)['model_state_dict'])
    model = model.to(device)

    gene_node_expression = test_data_loader.get_node_expression()
    prior_graph_topology = test_data_loader.get_prior_graph_topology()
    gene_node_expression = gene_node_expression.to(device)
    prior_graph_topology = prior_graph_topology.to(device)

    model.eval()
    output_list = []
    target_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_data_loader, ascii=True, disable=True)):
            graph_input = {'NODE': gene_node_expression,
                            'ENCODE': prior_graph_topology,
                            'DECODE': data['input']['EDGE'].transpose(0, 1).to(device)}
            branch_input = {'TF': data['input']['TF_feature'].to(device),
                            'TF_len': data['input']['TF_len'].to(device),
                            'GENE': data['input']['GENE_feature'].to(device)}

            target = data['label'].to(device)
            output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = model(graph_input, branch_input)
            output = torch.softmax(output, dim=1)[:, 1]                 
            output_list.append(output)
            target_list.append(target)

            if batch_idx % 1000 == 0:
                info = '[{}/{} ({:.0f}%)]'.format(batch_idx, len(test_data_loader), 100.0 * batch_idx / len(test_data_loader))
                logger.debug('Testing: ' + info)
                logger.debug(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    output = torch.cat(output_list, dim=0).detach().cpu().numpy()
    target = torch.cat(target_list, dim=0).detach().cpu().numpy()
    log = {met.__name__: met(output > 0.5, target) for _, met in enumerate(metric_ftns)}

    fpr_list, tpr_list, auroc = calculate_auroc(output, target)
    precision_list, recall_list, aupr = calculate_aupr(output, target)
    plot_roc_curve(fpr_list, tpr_list, os.path.join(config.result_dir, 'roc_curve.png'))
    plot_pr_curve(precision_list, recall_list, os.path.join(config.result_dir, 'pr_curve.png'))
    ep, epr = calculate_ep(output, target)
    log.update({'auroc': auroc, 'aupr': aupr, 'ep': ep, 'epr': epr})
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-m', '--mode', default='train', type=str, help='execution mode (train, test) (default: train)')
    parser.add_argument('-c', '--config', default='./config/debug_deepmix.json', type=str, help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--run_id', default=None, type=str, help='run id (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    customargs_tuple = collections.namedtuple('CustomArgs', ['flags', 'type', 'target'])
    customargs_options = [
        customargs_tuple(flags=['-seed', '--seed'], type=int, target='seed'),
        customargs_tuple(flags=['-lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        customargs_tuple(flags=['-bs', '--batch_size'], type=int, target='loader;args;batch_size'),
        customargs_tuple(flags=['-cell', '--cell'], type=int, target='loader;args;train_cell_number'),
        customargs_tuple(flags=['-size', '--size'], type=float, target='loader;args;train_data_size')
    ]
    for opt in customargs_options:
        parser.add_argument(*opt.flags, default=None, type=opt.type)

    args = parser.parse_args()
    config = ConfigParser.from_args(args, customargs_options)
    if args.cell is not None:
        config._config['model']['args']['graph_model_args']['input_dim'] = int(args.cell)

    if args.mode == 'train':
        train(config)
    elif args.mode == 'test':
        test(config)
    else:
        raise Exception('mode should be train or test')
    