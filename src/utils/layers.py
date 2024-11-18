import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_l2(outputs, labels):
    return F.cross_entropy(input=outputs, target=labels)

def loss_kl_divergence(outputs, labels, teacher_outputs, alpha, temperature):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/temperature, dim=1),
                             F.softmax(teacher_outputs/temperature, dim=1)) * (alpha * temperature * temperature) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

# student/teachers output shape : (N, 1000), labels shape : (N, )
def loss_kl_divergence_with_logits(student_output, labels, combined_teacher_output, temperature):
    z_t = combined_teacher_output
    z_s = student_output
    z_t_c = z_t - z_t[torch.arange(len(labels)), labels].unsqueeze(1)
    z_s_c = z_s - z_s[torch.arange(len(labels)), labels].unsqueeze(1)
    z = torch.min(z_t_c, z_s_c)
    
    p = F.softmax(z/temperature)
    p_t = F.softmax(z_t/temperature)
    p_s = F.softmax(z_s/temperature)
    
    kl_s = temperature**2 * nn.KLDivLoss()(p, p_s)
    kl_t = temperature**2 * nn.KLDivLoss()(p, p_t)
    
    return kl_s + kl_t

def get_combined_teacher_output(teacher_1_output, teacher_2_output, pi1, pi2):
    return pi1*teacher_1_output + pi2*teacher_2_output