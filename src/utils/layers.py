import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_l2(outputs, labels):
    return nn.MSELoss()(outputs, labels)

def loss_kl_divergence(outputs, labels, teacher_outputs, alpha, temperature):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/temperature, dim=1),
                             F.softmax(teacher_outputs/temperature, dim=1)) * (alpha * temperature * temperature) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

