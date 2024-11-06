import logging
import os

import torch
import torch.optim as optim
import random
import numpy as np

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model, load_model_bayes, load_model_and_build_BNN
from robustbench.utils import clean_accuracy as accuracy
from method import *
from method import rmt
from conf import cfg, load_cfg_fom_args

logger = logging.getLogger(__name__)

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def evaluate(description):
    load_cfg_fom_args(description)
    logger.info("test-time adaptation: {}".format(cfg.MODEL.ADAPTATION))
    #######################################################################
    if cfg.MODEL.ADAPTATION == "vcotta":
        base_model = load_model_and_build_BNN(cfg.MODEL.PRETRAIN, cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
        model = setup_vcotta(base_model)

    
    #######################################################################
    # evaluate on each severity and type of corruption in turn

    for severity in cfg.CORRUPTION.SEVERITY:

        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # continual adaptation for all corruption 
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
            err = 1. - acc


def setup_vcotta(model):
    model = vcotta.configure_model(model)
    params, param_names = vcotta.collect_bayes_params(model)
    optimizer = setup_optimizer(params)
    _model = vcotta.Vcotta(model, optimizer,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC, 
                        mt_alpha=cfg.OPTIM.MT, 
                        rst_m=cfg.OPTIM.RST, 
                        ap=cfg.OPTIM.AP)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return _model
#######################################################################

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    evaluate('"CIFAR-10-C evaluation.')
