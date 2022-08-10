"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
Fully-Connected implementation by Pouya Shiri @ pouyashiri.
"""
import sys

sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import pickle
from dataset_loader import *


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()
    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]

    return shifted_image.float()



class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3, fc_input=262144, num_pc=4):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules
        self.num_pc = num_pc

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.Linear(fc_input, self.num_pc * 8)
        #     self.capsules = nn.ModuleList(
        # [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
        #  range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:

            a = x[None, :, :, None, :]
            # print(f'a size = {a.size()}')
            b = self.route_weights[:, None, :, :, :]
            # print(f'b size = {b.size()}')
            priors = a @ b

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            # outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            # outputs = torch.cat(outputs, dim=-1)

            outputs = self.capsules(x.view(x.size(0), -1)).view(x.size(0), -1, 8)
            outputs = self.squash(outputs)

        return outputs


class CapsNetDecoder(nn.Module):
    def __init__(self, org_w, nc, nc_recon, type):
        super(CapsNetDecoder, self).__init__()
        self.type = type
        self.w = 10 if org_w == 32 else 6
        self.nc = nc

        if type == 'FC':
            # updated decoder to get both output vectors
            self.decoder = nn.Sequential(
                nn.Linear(16 * nc, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, org_w ** 2 * nc_recon),
                nn.Sigmoid()
            )
        else:
            self.fc_end = nn.Linear(16, 8 * (self.w ** 2))
            self.bn = nn.BatchNorm1d(8 * (self.w ** 2), momentum=0.8)
            self.deconvs = nn.Sequential(
                nn.ConvTranspose2d(8, 128, 3),
                nn.ConvTranspose2d(128, 64, 5),
                nn.ConvTranspose2d(64, 32, 5),
                nn.ConvTranspose2d(32, 16, 5),
                nn.ConvTranspose2d(16, 16, 5),
                nn.ConvTranspose2d(16, 16, 3),
                nn.ConvTranspose2d(16, nc_recon, 3))

    def forward(self, x, y, classes):
        if self.type == 'FC':
            if y is None:
                # In all batches, get the most active capsule.
                _, max_length_indices = classes.max(dim=1)
                y = Variable(torch.eye(self.nc)).cuda().index_select(dim=0, index=max_length_indices.data)
            reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))
        else:
            if y is None:
                _, max_length_indices = classes.max(dim=1)
            else:
                _, max_length_indices = y.max(dim=1)

            reconRes = torch.Tensor(np.zeros((x.size(0), x.size(2)))).cuda()
            for i in range(x.size(0)):
                reconRes[i, :] = x[i, max_length_indices[i], :]

            x = self.bn(self.fc_end(reconRes))

            x = x.reshape(x.size(0), 8, self.w, self.w)

            x = F.relu(self.deconvs(x))
            reconstructions = x.reshape(x.size(0), -1)

        return classes, reconstructions
    

class CapsuleNet(nn.Module):
    def __init__(self, num_class, niter, width, fc_input=262144, num_pc=4, in_channels=1, dec_type='FC'):
        super(CapsuleNet, self).__init__()

        self.nc = num_class
        self.fc_input = fc_input
        self.num_pc = num_pc
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=9, stride=1)
        w = (width - 9)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2, num_iterations=niter, num_pc=self.num_pc,
                                             fc_input=self.fc_input)
        w = int((w - 9) / 2 + 1)
        self.digit_capsules = CapsuleLayer(num_capsules=num_class, num_route_nodes=self.num_pc, in_channels=8,
                                           out_channels=16, num_iterations=niter, num_pc=self.num_pc,
                                           fc_input=self.fc_input)

        self.decoder = CapsNetDecoder(width, num_class, 1, dec_type)

    def forward(self, x, y=None):
        x = x.float()
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        classes, reconstructions = self.decoder(x, y, classes)

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self, hard, nc_recon):
        super(CapsuleLoss, self).__init__()
        self.hard = hard
        self.nc_recon = nc_recon
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        if self.hard:
            m_plus = 0.95
            m_minus = 0.05
        else:
            m_plus = 0.9
            m_minus = 0.1
            
        left = F.relu(m_plus - classes, inplace=True) ** 2
        right = F.relu(classes - m_minus, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        if self.nc_recon == 1:
            images = torch.sum(images, dim=1) / 3
            
        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST, FashionMNIST
    from torchvision.datasets.svhn import SVHN
    from torchvision.datasets.cifar import CIFAR10

    from tqdm import tqdm
    import torchnet as tnt
    import argparse, os, numpy

    parser = argparse.ArgumentParser(description="torchCapsNet.")
    parser.add_argument('--dset', default='mnist', required=True)
    parser.add_argument('--nc', default=10, type=int, required=True)
    parser.add_argument('--w', default=28, type=int, required=True)
    parser.add_argument('--npc', default=4, type=int, required=True)
    parser.add_argument('--dpath', default='')
    parser.add_argument('--bsize', default=128, type=int)
    parser.add_argument('--ich', default=1, type=int, required=True)

    parser.add_argument('--ne', default=50, type=int)
    parser.add_argument('--niter', default=3, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--hard', default=0, type=int, required=True)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--dec_type', default='FC')
    parser.add_argument('--nc_recon', default=1)
    parser.add_argument('--res_folder', default='output', required=True)
    parser.add_argument('--test_only', default=0, type=int)
    
    args = parser.parse_args()

    if not (os.path.exists(args.res_folder)):
        os.mkdir(args.res_folder)

    outputFolder = f'{args.dset}-{args.npc}'
    if args.hard == 1:
        outputFolder += '-H'
    expNoFile = f'{args.res_folder}/{outputFolder}/no'
    newExp = 1
    if os.path.exists(expNoFile):
        with open(expNoFile, 'r') as file:
            newExp = int(file.readline().replace('\n', '')) + 1
    else:
        if not os.path.exists(f'{args.res_folder}/{outputFolder}'):
            os.makedirs(f'{args.res_folder}/{outputFolder}')

    with open(expNoFile, 'w') as file:
        file.write(f'{newExp}\n')

    outputFolder = f'{args.res_folder}/{outputFolder}/{newExp}'
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)


    fc_input = ((args.w - 8) ** 2) * 256
    model = CapsuleNet(args.nc, args.niter, fc_input=fc_input, width=args.w, num_pc=args.npc, in_channels=args.ich, dec_type=args.dec_type)
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    if args.checkpoint != '':
        model.load_state_dict(torch.load(args.checkpoint))
        
    model.cuda()

    numparams = sum(param.numel() for param in model.parameters())
    # print("# parameters:", sum(param.numel() for param in model.parameters()))
    print(f'### PARAMS = {numparams}\n')
    
    optimizer = Adam(model.parameters(),lr=args.lr)

    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(args.nc, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(args.nc)),
                                                     'rownames': list(range(args.nc))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    train_loss = []
    train_error = []
    test_loss = []
    test_accuracy = []

    if args.hard == 0:
        capsule_loss = CapsuleLoss(hard=False, nc_recon=args.nc_recon)

    else:
        capsule_loss = CapsuleLoss(hard=True, nc_recon=args.nc_recon)

    tr_ep_times = []
    tr_time_temp = 0
    tst_times = []
    final_acc = 0


    def processor(sample):
        data, labels, training = sample
        if (len(data.size()) == 3):
            data = data.unsqueeze(1)
        # print(f'#######SIZE = {data.size()}')
        # data = augmentation(data.float() / 255.0)
        
        if args.dset == 'aff_expanded':
            labels = labels.reshape(labels.size(0))
            
        labels = torch.LongTensor(labels)

        labels = torch.eye(args.nc).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        # confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())


    import time


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])
        global tr_time_temp
        tr_time_temp = time.time()


    def on_end_epoch(state):
        tr_ep_times.append(time.time() - tr_time_temp)
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss.append(meter_loss.value()[0])
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error.append(meter_accuracy.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        temp = time.time()
        engine.test(processor, get_iterator(args.dset, args.bsize, False))
        tst_times.append(time.time() - temp)
        test_loss.append(meter_loss.value()[0])
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy.append(meter_accuracy.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        my_lr_scheduler.step()

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        global final_acc
        final_acc = meter_accuracy.value()[0]

        if int(state['epoch']) == args.ne:
            # torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])
            torch.save(model.state_dict(), f'{outputFolder}/checkpoint.pt')

        # Reconstruction visualization.

        # test_sample = next(iter(get_iterator(False)))

        # a = test_sample
        # if (len(a[0].size()) == 3):
        #     a[0] = a[0].unsqueeze(1)

        # ground_truth = (a[0].float() / 255.0)
        # _, reconstructions = model(Variable(ground_truth).cuda())
        # reconstruction = reconstructions.cpu().view_as(ground_truth).data

        # ground_truth_logger.log(
        #     make_grid(ground_truth, nrow=int(args.bsize ** 0.5), normalize=True, range=(0, 1)).numpy())
        # reconstruction_logger.log(
        #     make_grid(reconstruction, nrow=int(args.bsize ** 0.5), normalize=True, range=(0, 1)).numpy())


    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    if args.test_only == 1:
        state = engine.test(processor, get_iterator(args.dset, args.bsize, False))
        print(f"Test Accuracy: {meter_accuracy.value()[0]}")
        quit()
        
    engine.train(processor, get_iterator(args.dset, args.bsize, True), maxepoch=args.ne, optimizer=optimizer)

    with open(f'{outputFolder}/res', 'w') as file:
        file.write(f'{final_acc}\n{np.mean(tr_ep_times)}\n{np.mean(tst_times)}\n{numparams}')

    with open(f'{outputFolder}/det', 'w') as outFile:
        outFile.write('#epoch,tr_acc,tst_acc,tr_loss,tst_loss\n')
        for i in range(len(train_loss)):
            outFile.write(f'{i + 1},{train_error[i]},{test_accuracy[i]},{train_loss[i]},{test_loss[i]}\n')

    if args.checkpoint == '':
        import os
        hard_cmd = f'python main.py --dset {args.dset} --w {args.w} --nc {args.nc} --ich {args.ich} --npc {args.npc} --res_folder {args.res_folder} --hard 1 --checkpoint {outputFolder}/checkpoint.pt'
        with open('hardrun', 'w') as file:
            file.write(f'{hard_cmd}')
            
            
