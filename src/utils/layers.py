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
    # print(z_t)
    z_s = student_output
    # print(z_s)
    z_t_c = z_t - z_t[torch.arange(len(labels)), labels].unsqueeze(1)
    z_s_c = z_s - z_s[torch.arange(len(labels)), labels].unsqueeze(1)
    # print("------ z_t_c and z_t_s -------")
    # print(z_t_c)
    # print(z_s_c)
    
    z = torch.min(z_t_c, z_s_c)
    
    # print(z)
    
    p = F.log_softmax(z/temperature, dim=-1)
    # p_t = F.log_softmax(z_t/temperature, dim=-1).detach()
    # p_s = F.log_softmax(z_s/temperature, dim=-1)
    p_t = F.softmax((z_t - z_t.max(dim=-1, keepdim=True).values) / temperature, dim=-1).detach()
    p_s = F.softmax((z_s - z_s.max(dim=-1, keepdim=True).values) / temperature, dim=-1)
    # print("------ p_t and p_s -------")
    # print(p)
    # print(p_t)
    # print(p_s)
    kl_s = temperature**2 * nn.KLDivLoss(reduction='batchmean')(p, p_s)
    kl_t = temperature**2 * nn.KLDivLoss(reduction='batchmean')(p, p_t)
    
    # print(kl_s)
    # print(kl_t)
    return kl_s + kl_t

def get_combined_teacher_output(teacher_1_output, teacher_2_output, pi1, pi2):
    return pi1*teacher_1_output + pi2*teacher_2_output