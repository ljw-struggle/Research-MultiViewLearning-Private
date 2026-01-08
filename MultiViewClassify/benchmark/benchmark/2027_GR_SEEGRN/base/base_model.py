# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters]) # params = sum([p.numel() for p in model.parameters()])
        return super().__str__() + '\n' + 'Trainable parameters: {} \n'.format(params)


