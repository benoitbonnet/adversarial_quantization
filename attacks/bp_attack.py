import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def softmax_cross_entropy_our(logits, labels, device):
    y_hot = torch.zeros(logits.shape[0],logits.shape[1]).to(device)
    y_hot = y_hot.scatter_(1,torch.unsqueeze(labels,1),1)
    #y_hot = torch.autograd.Variable(y_hot, requires_grad=False)
    tmp = logits*y_hot
    logits_1 = logits - tmp
    j_best,_ = tmp.max(1)
    j_best_v = j_best.view(-1,1).expand(-1,logits.shape[1])
    logits_2 = logits_1 - j_best_v + y_hot*j_best_v
    tmp_s,_ = logits_2.max(1)
    up = tmp_s - j_best
    down = torch.log(torch.sum(torch.exp(logits_2)+1,1))
    loss = up - down

    return loss

def psi(grad_norm,delta_norm):
    cos_psi = torch.sum(grad_norm*delta_norm,[1,2,3])
    sin_psi = torch.sqrt(1-cos_psi.pow(2))
    tan_psi = sin_psi/cos_psi

    sin_psi = sin_psi.view(-1,1,1,1).expand(-1,grad_norm.shape[1],grad_norm.shape[2],grad_norm.shape[3])
    tan_psi = tan_psi.view(-1,1,1,1).expand(-1,grad_norm.shape[1],grad_norm.shape[2],grad_norm.shape[3])
    return tan_psi, sin_psi

def teddy_decay(i, steps, gamma):
    global_step = np.minimum(steps,i)
    rate = i/(steps+1.0)
    rage = 1- gamma
    epsi = gamma + rate*rage
    return epsi

def out_direction(d,g,normalized_grad,sng):
    lambd = torch.sum(d*normalized_grad/sng,[1,2,3])
    g_ort = d - lambd.view(-1,1,1,1).expand(-1,g.shape[1],g.shape[2],g.shape[3])*g
    return g_ort

def out_p(d,g,normalized_grad,snd,sng,sin_psi,beta,g_ort):
    flag = (beta == 0 )
    beta = torch.where(flag, torch.ones_like(beta), beta)
    mu = snd/beta*sin_psi -1
    flag = (mu <0).float()
    mu = (1-flag)*mu
    p = g_ort/(1+mu)
    return p

def in_p(d,normalized_delta,normalized_grad,beta,sin_psi,tan_psi,snd,sng,dt):
    p = d-normalized_grad*dt
    snp=p.view(d.shape[0],-1).norm(p=2,dim=1)
    flag = (snp<beta)
    cos_phi = beta.pow(2) - dt.pow(2) + snd.pow(2)
    cos_psi = cos_phi/(2*snd*beta)
    sin_phi = torch.sqrt(1-cos_phi.pow(2))
    p_o = ((cos_phi+sin_phi/tan_psi)*normalized_delta - (sin_phi/sin_psi)*normalized_grad)*beta

    p = torch.where(flag,p,p_o)
    return p

def Mdistortion(p,levels):
    pq = p#quantization(p,levels)
    dis_c = torch.sqrt(torch.sum(pq.pow(2),[1,2,3]))
    return dis_c.view(-1,1,1,1).expand(-1,p.shape[1],p.shape[2],p.shape[3])

def estimate_beta_out(d,g,normalized_grad,snd,sng,sin_psi,g_ort,dis,num_step,levels):
    ngo = g_ort.view(d.shape[0],-1).norm(p=2,dim=1)
    flag = (ngo==0)
    ngo = torch.where(flag,torch.ones_like(ngo),ngo)
    g_ort = g_ort/ngo.view(-1,1,1,1).expand(-1,g_ort.shape[1],g_ort.shape[2],g_ort.shape[3])
    tmp = torch.sum(d*g_ort,[1,2,3])
    tmp = tmp.view(-1,1,1,1).expand(-1,g_ort.shape[1],g_ort.shape[2],g_ort.shape[3])
    an_tmp = tmp**2 - d**2 + dis**2
    #dis = dis.view(-1,1,1,1).expand(-1,g_ort.shape[1],g_ort.shape[2],g_ort.shape[3])
    min_beta = tmp-torch.sqrt(an_tmp)
    max_beta = tmp

    p_min = min_beta*g_ort
    p_max = max_beta*g_ort

    DMin = Mdistortion(d-p_min,levels)
    DMax = Mdistortion(d-p_max,levels)

    for i in range(num_step):
        beta = (min_beta+max_beta)/2
        p = beta*g_ort
        D = Mdistortion(d-p,levels)

        flag = (D<dis)

        DMin = torch.where(flag,D,DMin)
        DMax = torch.where(flag,DMax,D)

        min_beta = torch.where(flag,beta,min_beta)
        max_beta = torch.where(flag,max_beta,beta)

    dMax = torch.abs(DMax-dis)
    dMin = torch.abs(DMin-dis)
    flag = (dMax<dMin)
    beta = torch.where(flag,max_beta,min_beta)

    return beta


def estimate_beta_in(d,g,snd,dis,device):
    ngo = g.view(d.shape[0],-1).norm(p=2,dim=1)
    g_ort = g/ngo.view(-1,1,1,1).expand(-1,g.shape[1],g.shape[2],g.shape[3])
    tmp = torch.sum(d*g_ort,[1,2,3])
    tmp = tmp.view(-1,1,1,1).expand(-1,g.shape[1],g.shape[2],g.shape[3])
    bac = dis.pow(2) - snd.pow(2) + tmp.pow(2)
    beta = tmp + torch.sqrt(bac)
    beta_min = 0.1*torch.ones(dis.shape)
    beta_max = dis - snd
    beta_min = beta_min.to(device)
    flag = (beta<beta_min)
    beta = torch.where(flag,beta_min,beta)

    return beta

def classif_loss(model, inputs_variable ,labels, num_classes, attack_device):
    batch_size = labels.shape[0]
    prediction = model(inputs_variable)
    labels_onehot = torch.zeros(batch_size, num_classes, device=attack_device)
    labels_onehot.scatter_(1, labels.unsqueeze(1).long(),1)

    adversarial_loss = prediction*labels_onehot
    adversarial_loss = adversarial_loss.sum(axis=1)
    adversarial_loss.backward(torch.ones(batch_size, device=attack_device))
    gradients = inputs_variable.grad.data
    return(prediction.argmax(1),gradients)

class BP:

    def __init__(self,
                steps=50,
                gamma = 0.3,
                upper_radius=1000,
                levels = 256,
                num_classes=1000,
                device = torch.device('cpu')):
        self.steps = steps
        self.gamma = gamma
        self.levels = levels
        self.device = device
        self.upper_radius = upper_radius
        self.num_classes = num_classes

    def attack(self,model,inputs,labels,targeted=False):
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        adv = torch.ones_like(inputs, requires_grad=True)*inputs

        flag_cross = torch.zeros(inputs.shape[0], dtype=torch.uint8)
        flag_cross = torch.ByteTensor(flag_cross)
        flag_cross = flag_cross.to(self.device)

        best_x = torch.zeros_like(inputs)
        for i in range(self.steps):
            delta = inputs - adv.clone()
            delta_norm = delta.data.view(batch_size, -1).norm(p=2, dim=1)
            delta_norm = delta_norm.view(batch_size,1,1,1).expand(delta.shape)
            flag = (delta_norm==0)
            delta_norm = torch.where(flag, torch.ones_like(delta_norm), delta_norm)
            normalized_delta = delta/delta_norm
            adv = torch.autograd.Variable(adv, requires_grad=True)

            pred_labels, grad = classif_loss(model, adv, labels, self.num_classes, self.device)
            grad_norm = grad.view(batch_size,-1).norm(dim=1).view(batch_size,1,1,1)
            normalized_grad = - grad/grad_norm

            # not implement yet
            tan_psi, sin_psi = psi(normalized_grad, normalized_delta)
            eps = teddy_decay(i,self.steps, self.gamma)

            # stage 1
            p_search = normalized_grad * 1

            # stage 2
            # out
            g_ort = out_direction(delta,grad,normalized_grad,grad_norm)
            dis = delta_norm*eps
            beta =estimate_beta_out(delta,grad,normalized_grad,delta_norm,grad_norm,sin_psi,g_ort,dis,7,self.levels)
            beta_out = beta
            p_out = out_p(delta,grad,normalized_grad,delta_norm,grad_norm,sin_psi,beta_out,g_ort)
            # in
            dis = delta_norm/eps
            beta = estimate_beta_in(delta,grad,delta_norm,dis,self.device)
            beta_in =beta
            p_in = normalized_grad*beta

            is_adv = torch.ByteTensor(pred_labels == labels) if targeted else (pred_labels != labels).view(batch_size,1,1,1)
            flag_cross = flag_cross + is_adv

            fc = flag_cross.view(batch_size,1,1,1).expand(grad.shape)

            delta = torch.where(fc, p_in, p_search)
            delta = torch.where(is_adv, p_out,delta)

            adv = torch.clamp(adv+delta,0,255)
            #adv = quantization(adv,self.levels)
            logits = model(adv)
            pred_labels = logits.argmax(1)
            is_adv = (pred_labels == labels) if targeted else (pred_labels != labels).view(batch_size,1,1,1)

            a = (best_x - inputs).view(batch_size, -1).norm(p=2,dim=1)
            b = (adv - inputs).view(batch_size, -1).norm(p=2,dim=1)

            flag_save =(a>b).view(batch_size,1,1,1).expand(-1,grad.shape[1],grad.shape[2],grad.shape[3])
            nm_best_x = torch.where(flag_save, adv,best_x)
            best_x = torch.where(is_adv, nm_best_x,best_x)
            model.zero_grad()

        return best_x.detach()
