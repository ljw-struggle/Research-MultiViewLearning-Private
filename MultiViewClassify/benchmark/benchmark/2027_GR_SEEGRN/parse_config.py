# -*- coding: utf-8 -*-
import os
import torch
import random
import logging
import numpy as np

from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json, create_dirs


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = self.update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        exper_name = self.config['name']
        save_dir = self.config['save_dir']
        run_id = datetime.now().strftime(r'%m%d_%H%M%S') if run_id is None else run_id # use timestamp as default run-id
        self._result_dir = os.path.join(save_dir, exper_name, str(run_id))
        self._model_dir = os.path.join(self.result_dir, 'model')
        self._log_dir = os.path.join(self.result_dir, 'log')
        assert not os.path.exists(self.result_dir), "Result directory already exists. Please delete or rename the existing directory."
        create_dirs([self.result_dir, self.model_dir, self.log_dir])
        write_json(self.config, os.path.join(self.model_dir, 'config.json'))

        # set random seed for reproducibility
        self.set_reproducibility(self.config['seed'])

        # set device
        self.device, self.device_ids = self.prepare_device(self.config['n_gpu'])

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    @classmethod
    def from_args(cls, args, options):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if args.config is None and args.resume is None:
            raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

        args_config_fname = None
        resume_config_fname = None
        if args.config is not None:
            args_config_fname = args.config
        if args.resume is not None:
            resume_parent_dir = os.path.dirname(args.resume)
            resume_config_fname = os.path.join(resume_parent_dir, 'config.json')
        
        if args_config_fname is not None and resume_config_fname is not None:
            # update new config for fine-tuning
            args_config = read_json(args_config_fname)
            resume_config = read_json(resume_config_fname)
            resume_config.update(args_config)
            config = resume_config
        
        if args_config_fname is not None and resume_config_fname is None:
            args_config = read_json(args_config_fname)
            config = args_config

        if args_config_fname is None and resume_config_fname is not None:
            resume_config = read_json(resume_config_fname)
            config = resume_config

        # parse custom cli options into dictionary
        get_opt_name = lambda flags: [flag.replace('--', '') for flag in flags if flag.startswith('--')][0]
        modification = {opt.target : getattr(args, get_opt_name(opt.flags)) for opt in options}       
        return cls(config, args.resume, modification, args.run_id)

    @classmethod
    def update_config(cls, config, modification):
        # helper functions to update config dict with custom cli options
        for k, v in modification.items():
            if v is not None:
                keys = k.split(';')
                reduce(getitem, keys[:-1], config)[keys[-1]] = v
        return config

    def get_logger(self, name, verbosity=2):
        assert verbosity in self.log_levels, 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def set_reproducibility(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

    def prepare_device(self, n_gpu_use):
        """
        setup GPU device if available. get gpu device indices which are used for DataParallel
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine,"
                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
                "available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        # assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        # assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def result_dir(self):
        return self._result_dir

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def log_dir(self):
        return self._log_dir
