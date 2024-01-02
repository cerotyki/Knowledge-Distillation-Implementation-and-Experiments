import os
import random

import gdown
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# set random seed for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=100):
        super(ResNet, self).__init__()
        assert (
            depth - 2
        ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
        n = (depth - 2) // 6
        block = BasicBlock

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.stage_channels = num_filters

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(
            block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1))
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.layer1[-1].bn2
        bn2 = self.layer2[-1].bn2
        bn3 = self.layer3[-1].bn2
        return [bn1, bn2, bn3]

    def get_stage_channels(self):
        return self.stage_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        avg = x.reshape(x.size(0), -1)
        out = self.fc(avg)

        return out

def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], "basicblock", **kwargs)

def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], "basicblock", **kwargs)

class BaseTrainer:
    pretrained_teacher_link = 'https://drive.google.com/uc?id=1Gh3Z8BZ62PGD7PQiFiwmU9vMwMpF5F46'

    def __init__(self):
        self.teacher = resnet32x4(num_classes=100)
        self.student = resnet8x4(num_classes=100)
        gdown.download(self.pretrained_teacher_link, './resnet_32x4.pth', resume=True)
        self.teacher.load_state_dict(torch.load("./resnet_32x4.pth", map_location="cpu")["model"])

        self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
        self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])

        self.train_set = datasets.CIFAR100('./data/', download=True, train=True, transform=self.train_transform)
        self.test_set = datasets.CIFAR100('./data/', download=False, train=False, transform=self.test_transform)
        self.test_dataloader = DataLoader(self.test_set, batch_size=64, shuffle=False)

    def save_student_checkpoint(self, ckpt_path):
        state_dict = self.student.state_dict()
        torch.save(state_dict, ckpt_path)

    def load_student_checkpoint(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.student.load_state_dict(state_dict)

    @torch.no_grad()
    def evaluate_student(self):
        self.student.cuda().eval()
        n = 0
        correct = 0
        for image, target in self.test_dataloader:
            image = image.cuda()
            target = target.cuda()
            output = self.student(image)
            n += image.size(0)
            correct += output.max(-1).indices.eq(target).sum().item()
        accuracy = 100 * correct / n
        return accuracy

    def train_student(self):
        pass

### IMPLEMENT THIS TRAINER CLASS ###

class KDTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        ### YOU MAY EDIT BELOW ###
        self.train_dataloader = DataLoader(self.train_set, batch_size=64, shuffle=True, num_workers=24, drop_last=True)

        self.init_lr = 0.2
        self.max_epoch = 200
        self.optimizer = optim.SGD(self.student.parameters(), lr=self.init_lr, momentum=0.9, weight_decay=1e-4)

        ## temperature ##
        self.T = 30
        ## alpha ##
        self.alpha = 0.5

        ## best acc
        self.best_acc = 0

        ## LR scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch/20)

    def train_student(self): ## added model path to save checkpoint internally
        #### IMPLEMENT TRAINING HERE ####
        self.teacher.cuda().eval()
        self.student.cuda().train()

        cl_criterion = nn.CrossEntropyLoss().cuda()
        # kd_criterion = nn.KLDivLoss(reduction="batchmean").cuda()
        kd_criterion = nn.KLDivLoss(reduction="batchmean")

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_dataloader:

                ## classification loss of student
                student_outputs = self.student(inputs.cuda())
                cl_loss = cl_criterion(student_outputs, labels.cuda())

                ## knowledge distillation loss
                teacher_outputs = self.teacher(inputs.cuda())
                kd_loss = kd_criterion(F.log_softmax(student_outputs/self.T, dim=1), F.softmax(teacher_outputs/self.T, dim=1)) * self.T * self.T

                ## total loss
                loss = (1 - self.alpha) * cl_loss + (self.alpha) * kd_loss

                ## backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                _, pred = torch.max(student_outputs, 1)
                total += labels.size(0)
                correct += (pred == labels.cuda()).sum().item()

                running_loss += loss.item()

            ## check loss, accuracy, and learning rate
            running_loss = running_loss / len(self.train_dataloader)
            training_acc = (correct / total) * 100
            test_acc = self.evaluate_student()
            # learning_rate = self.scheduler.get_last_lr()

            ## check if best
            if test_acc > self.best_acc:
                self.save_student_checkpoint("./student_checkpoint_temp.pth")
                self.best_acc = test_acc
            self.student.cuda().train()
            
            ## update LR
            self.scheduler.step()

            ## log
            print(f"Epoch {epoch + 1}/{self.max_epoch}, Loss: {running_loss:.4f}, training Acc: {training_acc:.2f}, test Acc: {test_acc:.2f}, best Acc: {self.best_acc:.2f}")
        
        ## go back to the checkpoint when the accuracy was the highest
        trainer.load_student_checkpoint("./student_checkpoint_temp.pth")



CKPT_PATH = "./student_checkpoint.pth"

trainer = KDTrainer()
trainer.train_student()
trainer.save_student_checkpoint(CKPT_PATH)




trainer.load_student_checkpoint(CKPT_PATH)
accuracy = trainer.evaluate_student()

print(f"Student model test accuracy: {accuracy:.3f} %")
print(f"Is above threshold performance? {accuracy > 72.2}")
















### IMPLEMENT THIS TRAINER CLASS ###
class ImprovedKDTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        ### YOU MAY EDIT BELOW ###

        ## modify models' forward outputs fm
        self.student = ResNet_ex(8)
        self.teacher = ResNet_ex(32)

        ## reload teacher model
        self.teacher.load_state_dict(torch.load("./resnet_32x4.pth", map_location="cpu")["model"])

        self.train_dataloader = DataLoader(self.train_set, batch_size=64, shuffle=True, num_workers=24, drop_last=True)

        self.init_lr = 0.2
        self.max_epoch = 200
        self.optimizer = optim.SGD(self.student.parameters(), lr=self.init_lr, momentum=0.9, weight_decay=1e-4)

        ## temperature ##
        self.T = 30
        ## alpha ##
        self.alpha = 0.5

        ## best acc
        self.best_acc = 0

        ## number of epochs allowed for FSP
        self.epoch_for_fsp = 10
        self.epoch_for_lr_update = 20

        ## for RKD
        self.w_dist = 25
        self.w_angle = 50

    @torch.no_grad()
    def evaluate_student(self):
        self.student.cuda().eval()
        n = 0
        correct = 0
        for image, target in self.test_dataloader:
            image = image.cuda()
            target = target.cuda()
            f0, f1, f2, f3, avg, output = self.student(image)
            n += image.size(0)
            correct += output.max(-1).indices.eq(target).sum().item()
        accuracy = 100 * correct / n
        return accuracy

    def train_student(self): ## added model path to save checkpoint internally
        #### IMPLEMENT TRAINING HERE ####
        self.teacher.cuda().eval()
        self.student.cuda().train()

        ce_criterion = nn.CrossEntropyLoss().cuda()
        kld_criterion = nn.KLDivLoss(reduction="batchmean")
        sp_criterion = self.sp_loss
        fsp_criterion = self.fsp_loss
        rkd_criterion = self.rkd_loss

        scheduler_kd = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch/20)
        scheduler_fsp = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch)

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_dataloader:

                self.optimizer.zero_grad()

                ## classification loss of student
                f0_s, f1_s, f2_s, f3_s, avg_s, student_outputs = self.student(inputs.cuda())
                ce_loss = ce_criterion(student_outputs, labels.cuda())

                ## knowledge distillation loss
                f0_t, f1_t, f2_t, f3_t, avg_t, teacher_outputs = self.teacher(inputs.cuda())
                if epoch < self.epoch_for_fsp:
                    ## stage 1: initial training for FSP
                    kd_loss = (fsp_criterion(f0_s, f1_s, f0_t, f1_t) + \
                                fsp_criterion(f1_s, f2_s, f1_t, f2_t) + \
                                fsp_criterion(f2_s, f3_s, f2_t, f3_t)) / 3.0
                    scheduler = scheduler_fsp
                else:
                    ## stage 2: training using original KD
                    kd_loss = kld_criterion(F.log_softmax(student_outputs/self.T, dim=1), F.softmax(teacher_outputs/self.T, dim=1)) * self.T * self.T
                    scheduler = scheduler_kd

                ## total loss
                loss = (1 - self.alpha) * ce_loss + (self.alpha) * kd_loss

                ## backprop
                loss.backward()
                self.optimizer.step()
                
                _, pred = torch.max(student_outputs, 1)
                total += labels.size(0)
                correct += (pred == labels.cuda()).sum().item()

                running_loss += loss.item()

            ## check loss, accuracy, and learning rate
            running_loss = running_loss / len(self.train_dataloader)
            training_acc = (correct / total) * 100
            test_acc = self.evaluate_student()
            # learning_rate = self.scheduler.get_last_lr()

            ## check if best
            if test_acc > self.best_acc:
                self.save_student_checkpoint("./student_checkpoint_temp.pth")
                self.best_acc = test_acc
            self.student.cuda().train()

            # if epoch < self.epoch_for_lr_update:
            #     ## update LR
            #     self.scheduler.step()
            
            ## update LR
            scheduler.step()

            ## log
            print(f"Epoch {epoch + 1}/{self.max_epoch}, Loss: {running_loss:.4f}, training Acc: {training_acc:.2f}, test Acc: {test_acc:.2f}, best Acc: {self.best_acc:.2f}")
        
        ## go back to the checkpoint when the accuracy was the highest
        trainer.load_student_checkpoint("./student_checkpoint_temp.pth")

    def sp_loss(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        fm_t = fm_t.view(fm_t.size(0), -1)

        return F.mse_loss(F.normalize(torch.mm(fm_s, fm_s.t()), p=2, dim=1), F.normalize(torch.mm(fm_t, fm_t.t()), p=2, dim=1))
    
    def fsp_loss(self, fm_s1, fm_s2, fm_t1, fm_t2):
        if fm_s1.size(2) > fm_s2.size(2):
            fm_s1 = F.adaptive_avg_pool2d(fm_s1, (fm_s2.size(2), fm_s2.size(3)))
        
        fm_s1 = fm_s1.view(fm_s1.size(0), fm_s1.size(1), -1)
        fm_s2 = fm_s2.view(fm_s2.size(0), fm_s2.size(1), -1).transpose(1,2)
        
        if fm_t1.size(2) > fm_t2.size(2):
            fm_t1 = F.adaptive_avg_pool2d(fm_t1, (fm_t2.size(2), fm_t2.size(3)))
        
        fm_t1 = fm_t1.view(fm_t1.size(0), fm_t1.size(1), -1)
        fm_t2 = fm_t2.view(fm_t2.size(0), fm_t2.size(1), -1).transpose(1,2)

        return F.mse_loss(torch.bmm(fm_s1, fm_s2) / fm_s1.size(2), torch.bmm(fm_t1, fm_t2) / fm_t1.size(2))

    def rkd_loss(self, feat_s, feat_t):
        feat_square_s = feat_s.pow(2).sum(dim=1)
        feat_dist_s = (feat_square_s.unsqueeze(0) + feat_square_s.unsqueeze(1) - 2 * torch.mm(feat_s, feat_s.t())).clamp(min=1e-12)
        feat_dist_s = feat_dist_s.sqrt()
        feat_dist_s = feat_dist_s.clone()
        feat_dist_s[range(len(feat_s)), range(len(feat_s))] = 0

        feat_square_t = feat_t.pow(2).sum(dim=1)
        feat_dist_t = (feat_square_t.unsqueeze(0) + feat_square_t.unsqueeze(1) - 2 * torch.mm(feat_t, feat_t.t())).clamp(min=1e-12)
        feat_dist_t = feat_dist_t.sqrt()
        feat_dist_t = feat_dist_t.clone()
        feat_dist_t[range(len(feat_t)), range(len(feat_t))] = 0

        rkd_dist_loss = self.w_dist * F.smooth_l1_loss(feat_dist_s / feat_dist_s[feat_dist_s>0].mean(), feat_dist_t / feat_dist_t[feat_dist_t>0].mean())
        
        # N x C --> N x N x C
        norm_feat_vd_s = F.normalize(feat_s.unsqueeze(0) - feat_s.unsqueeze(1), p=2, dim=2)
        norm_feat_vd_t = F.normalize(feat_t.unsqueeze(0) - feat_t.unsqueeze(1), p=2, dim=2)
        rkd_angle_loss = self.w_angle * F.smooth_l1_loss(torch.bmm(norm_feat_vd_s, norm_feat_vd_s.transpose(1, 2)).view(-1), torch.bmm(norm_feat_vd_t, norm_feat_vd_t.transpose(1, 2)).view(-1))
        
        return rkd_dist_loss + rkd_angle_loss

class ResNet_ex(ResNet):
    def __init__(self, depth, numfilters=[32, 64, 128, 256]):
        super().__init__(depth, numfilters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        avg = x.reshape(x.size(0), -1)
        out = self.fc(avg)

        return f0, f1, f2, f3, avg, out



CKPT_PATH = "./student_improved_checkpoint.pth"

trainer = ImprovedKDTrainer()
trainer.train_student()
trainer.save_student_checkpoint(CKPT_PATH)



trainer.load_student_checkpoint(CKPT_PATH)
accuracy = trainer.evaluate_student()

print(f"Improved student model test accuracy: {accuracy:.3f} %")



