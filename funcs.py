import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from utils import *
from modules import LabelSmoothing
import torch.distributed as dist
import random
import os
import time
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
from torch import autocast
from torch.cuda.amp import GradScaler

def add_dimention(x, T):
    x = x.unsqueeze(1).repeat(1, T, 1, 1, 1)

    return x

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.float().cuda(device)
            label = label.cuda(device)
            out = model(img).mean(0)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)
            tot += (label == out.max(1)[1]).sum().data
    return tot / length, epoch_loss / length


def train_ann_cifar10_dvs(train_dataloader, test_dataloader, optimizer, model, epochs, device, loss_fn, lr=0.1, wd=5e-4,
              parallel=False,
              rank=0):
    model.cuda(device)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        length = 0
        train_loss = 0.0
        train_acc = 0.0
        train_samples = 0
        model.train()
        for img, label in train_dataloader:
            img = img.float().cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            out= model(img).mean(0)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out.argmax(1) == label).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples

        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
        print('Epoch {} ->train_loss:{},train_acc:{}, Val_loss: {}, Acc: {}'.format(epoch, train_loss, train_acc,
                                                                                    val_loss, tmp_acc), flush=True)
        torch.save(model.state_dict(), './checkpoint_temp_ann_snn_l_16_dvs.pth')
        if rank == 0 and tmp_acc >= best_acc:
            torch.save(model.state_dict(), './saved_models_std_ann_snn_l_16_dvs/checkpoint_max_ann.pth')
        best_acc = max(tmp_acc, best_acc)
        print('best_acc: ', best_acc)
        scheduler.step()
    return best_acc, model



def train_snn(model, device, train_loader, test_loader, criterion, epochs, lr=0.1, wd=5e-4):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    max_test_acc = 0.0
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_loader:
            optimizer.zero_grad()
            frame = frame.float().to(device)
            label = label.to(device)
            out_fr = model(frame).mean(0)
            loss = F.cross_entropy(out_fr, label)
            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples

        scheduler.step()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_loader:
                frame = frame.float().to(device)
                label = label.to(device)
                out_fr = model(frame).mean(0)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(model)

        test_loss /= test_samples
        test_acc /= test_samples

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), "./dvscifar10/test1.pth")
            print("model saved")
            print("best_acc:{}".format(max_test_acc))
            print(
                f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')


def distillation_step_loss(y, student_scores_father, teacher_scores, student_scores, temperature):
    # 计算交叉熵损失
    loss_ce = F.cross_entropy(student_scores, y)

    # 计算蒸馏损失
    T = student_scores_father.size(0)
    loss_distillation = 0
    for t in range(T):
        loss_distillation += nn.KLDivLoss()(torch.log_softmax(student_scores_father[t, ...] / temperature, dim=1),
                                            torch.softmax(teacher_scores / temperature, dim=1)) * temperature ** 2
    loss_distillation = loss_distillation / T  # L_KLD

    # 综合损失
    loss = loss_ce + 1 * loss_distillation
    return loss


def train_snn_step(large_model, small_model, device, train_loader, test_loader, criterion, epochs, lr=0.1, wd=5e-4):
    temperature = 2

    optimizer = torch.optim.Adam(small_model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)
    max_test_acc=0.0

    for epoch in range(epochs):
        start_time = time.time()
        small_model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for frame, label in train_loader:
            optimizer.zero_grad()
            frame = frame.float().to(device)
            label = label.to(device)
            with torch.enable_grad():
                with torch.no_grad():
                    teacher_output = large_model(frame).mean(0)
                student_output = small_model(frame).mean(0)

                loss = distillation_step_loss(label, teacher_output, student_output, temperature)
                loss.backward()

            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (student_output.argmax(1) == label).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples

        scheduler.step()

        small_model.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_loader:
                frame = frame.float().to(device)
                label = label.to(device)
                # label_onehot = F.one_hot(label, 10).float()
                out_fr = small_model(frame).mean(0)
                # loss = F.mse_loss(out_fr, label_onehot)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(small_model)

        test_loss /= test_samples
        test_acc /= test_samples

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(small_model.state_dict(), "./ncal_model_weight/KL_t_1.pth")
            print("模型已保存")
            print("best_acc:{}".format(max_test_acc))
            print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')



def train_snn2(large_model, small_model, device, train_loader, test_loader, criterion, epochs, lr=0.1, wd=5e-4):
    temperature = 10

    optimizer = torch.optim.Adam(small_model.parameters(), lr=lr)

    #optimizer=torch.optim.SGD(small_model.parameters(),lr=lr,momentum=0.9)
    max_test_acc = 0.0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=epochs)
    for epoch in range(epochs):
        start_time = time.time()
        small_model.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_loader:
            optimizer.zero_grad()
            # frame1=add_dimention(frame,T=1)
            # frame1 = frame1.float().to(device)
            # frame2=add_dimention(frame,T=4)
            # frame2 = frame2.float().to(device)
            frame = frame.float().to(device)
            label = label.to(device)
            with torch.no_grad():
                teacher_output = large_model(frame).mean(0)
            student_output = small_model(frame).mean(0)
            student = small_model(frame)
            # loss = F.mse_loss(out_fr, label_onehot)
            # loss = distillation_TET_loss(label,student,teacher_output,student_output,temperature)
            loss = distillation_step_loss(label, student, teacher_output, student_output, temperature)  # 针对每一个时间步进行蒸馏
            loss.backward()
            optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (student_output.argmax(1) == label).float().sum().item()

        train_loss /= train_samples
        train_acc /= train_samples

        scheduler.step()

        small_model.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_loader:
                # frame2 = add_dimention(frame, T=4)
                # frame2 = frame2.float().to(device)
                frame = frame.float().to(device)
                label = label.to(device)
                out_fr = small_model(frame).mean(0)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(small_model)

        test_loss /= test_samples
        test_acc /= test_samples

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(small_model.state_dict(), "./dvscifar10_model_weight/KL_t_1.pth")
            print("模型已保存")
            print("best_acc:{}".format(max_test_acc))
            print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')

def eval_snn(test_dataloader, model, device, sim_len=8, rank=0):
    tot = torch.zeros(sim_len).cuda(device)
    length = 0
    model = model.cuda(device)
    model.eval()
    # valuate
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):
            spikes = 0
            length += len(label)
            img = img.float().cuda()
            label = label.cuda()
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label == spikes.max(1)[1]).sum()
            reset_net(model)
    return tot / length


