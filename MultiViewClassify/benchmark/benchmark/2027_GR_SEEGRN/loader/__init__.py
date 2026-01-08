# -*- coding: utf-8 -*-
from .loader_deepmix import *
from .loader_sequence import *


class loader_deepmix:
    Tissue_Specific_Loader = Tissue_Specific_Loader
    Cell_Type_Specific_Loader = Cell_Type_Specific_Loader


class loader_sequence:
    HomoSapiensLoader = HomosapiensLoader_sequence_pretrain
    MusMusculusLoader = MusmusculusLoader_sequence_pretrain
    CrossSpeciesLoader = CrossspeciesLoader_sequence_pretrain
