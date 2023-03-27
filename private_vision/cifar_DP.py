import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import pandas as pd
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from private_vision import PrivacyEngine
import opacus
from tqdm import tqdm
import warnings
import timm
import glob
import model

def prepare(args):
    device = torch.device("cuda")

    # Data
    print('==> Preparing data..')
    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(name)
                continue
            try:
                own_state[name].copy_(param.data)
            except KeyError:
                print(f"Skipping key '{name}' due to missing key error")


    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if args.cifar_data == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root='/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='/data', train=False, download=True, transform=transform_test)
    elif args.cifar_data == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(
            root='/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(
            root='/data', train=False, download=True, transform=transform_test)
    elif args.cifar_data == 'celeba':
        class private_dataset(Dataset):

            def __init__(self, image_loc, label_loc, transform):

                filenames = []
                for root, dirs, files in os.walk(image_loc):
                    for file in files:
                        if file.endswith('.jpg') == True or file.endswith('.png') == True:
                            filenames.append(file)

                self.full_filenames = glob.glob(image_loc+'*/*/*.*')

                label_df = pd.read_csv(label_loc)
                label_df.set_index("filename", inplace=True)
                self.labels = [label_df.loc[filename].values[0]
                               for filename in filenames]
                self.transform = transform

            def __len__(self):
                    return len(self.full_filenames)

            def __getitem__(self, idx):
                    image = Image.open(self.full_filenames[idx])
                    image = image.convert('RGB')
                    image = self.transform(image)
                    return image, self.labels[idx]

        def transformate(crop_size, re_size):
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
            def crop(x): return x[:, offset_height:offset_height +
                                  crop_size, offset_width:offset_width + crop_size]
            proc = []
            proc.append(transforms.ToTensor())
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip())
            proc.append(transforms.ToTensor())
            return transforms.Compose(proc)
        data_dir = 'E:/NCKH2023/LPG-MI/datasets/celeba_private_domain'
        label_loc = 'E:/NCKH2023/LPG-MI\data_files/train_label.csv'
        trainset = private_dataset(
            data_dir, label_loc, transformate(108, 64))
        trainloader = DataLoader(
            trainset, batch_size=args.mini_bs, num_workers=0, shuffle=True)

    if args.cifar_data != 'vgg16_bn':
        if args.cifar_data == 'CIFAR10':
            trainset = torchvision.datasets.CIFAR10(
            root='/data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
            root='/data', train=False, download=True, transform=transform_test)
        elif args.cifar_data == 'CIFAR100':
            trainset = torchvision.datasets.CIFAR100(
            root='/data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR100(
            root='/data', train=False, download=True, transform=transform_test)

            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=2)
            trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.mini_bs, shuffle=True, num_workers=2)

    print('==> Building model..', args.model, '  mode ', args.mode)
    NUM_CLASSES = 10 if args.cifar_data == 'CIFAR10' else 100
    if args.model != 'vgg16_bn':
        net = timm.create_model(
        args.model, pretrained=args.pretrained, num_classes=NUM_CLASSES)
    elif args.model == 'vgg16_bn':
        net = model.VGG16_bn(1000)
        BACKBONE_RESUME_ROOT = "E:/NCKH2023/LPG-MI/checkpoints/target_model/target_ckp/VGG_MixedGhost.tar"
        print("Loading Backbone Checkpoint ")
        checkpoint = torch.load(BACKBONE_RESUME_ROOT)
        net = ModuleValidator.fix(net)
        net.load_state_dict(checkpoint['state_dict'])

    net = ModuleValidator.fix(net)
    net.to(device)
    if 'xcit' in args.model:
        for name, param in net.named_parameters():
            if 'gamma' in name or 'attn.temperature' in name:
                param.requires_grad = False

    if 'cait' in args.model:
        for name, param in net.named_parameters():
            if 'gamma_' in name:
                param.requires_grad = False

    if 'convnext' in args.model:
        for name, param in net.named_parameters():
            if '.gamma' in name or 'head.norm.' in name or 'downsample.0' in name or 'stem.1' in name:
                param.requires_grad = False

    if 'convit' in args.model:
        for name, param in net.named_parameters():
            if 'attn.gating_param' in name:
                param.requires_grad = False

    if 'beit' in args.model:
        for name, param in net.named_parameters():
            if 'gamma_' in name or 'relative_position_bias_table' in name or 'attn.qkv.weight' in name or 'attn.q_bias' in name or 'attn.v_bias' in name:
                param.requires_grad = False

    for name, param in net.named_parameters():
        if 'cls_token' in name or 'pos_embed' in name:
            param.requires_grad = False

    print('number of parameters: ', sum([p.numel() for p in net.parameters()]))

    if "ghost" in args.mode:
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss()
    # if args.model =='vgg16_bn':
    #     optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # else :
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    n_acc_steps = args.bs // args.mini_bs

    if 'ghost' in args.mode:
        sigma = get_noise_multiplier(
            target_epsilon=args.eps,
            target_delta=1e-5,
            sample_rate=args.bs/len(trainset),
            epochs=args.epochs,
            accountant="rdp"
        )
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainloader.dataset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            max_grad_norm=args.grad_norm,
            ghost_clipping='non' not in args.mode,
            mixed='mixed' in args.mode
        )
        privacy_engine.attach(optimizer)
    # Training
    if args.model != 'vgg16_bn' :
        def train(epoch):
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc = 'Batch')):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                if args.mode == 'non-private':
                    loss.backward()
                    if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                        optimizer.step(loss=loss)
                        optimizer.zero_grad()
                    else:
                        optimizer.virtual_step(loss=loss)
                train_loss += loss.mean().item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        def test(epoch):
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    loss = loss.mean()
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                print(epoch, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return args.epochs, train, test

    if args.model == 'vgg16_bn' :
         # Training
        def train(epoch):
            local_loss = checkpoint['train_loss_min']
            for epoch in tqdm(range(epoch)):
                print('==> Building model..', args.model, '  mode ', 'mixed ghost clipping DP')
                print('\nEpoch: %d' % epoch)
                net.train()
                train_loss = 0
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader),desc ='Batch'):
                    inputs, targets = inputs.to(device), targets.to(device)
                    feat, outputs = net(inputs)
                    loss = criterion(outputs, targets)


                    if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                            optimizer.step(loss=loss)
                            optimizer.zero_grad()
                    else:
                            optimizer.virtual_step(loss=loss)
                    train_loss += loss.mean().item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # writer.add_scalar('training loss',
                    #                  train_loss/(batch_idx+1),
                    #                  epoch * len(trainloader) + batch_idx)
                print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                print(
                        'ε = {} '.format(args.eps)
                        )

                avg_train_loss = train_loss/(batch_idx+1)

                if avg_train_loss <= local_loss :
                        print('Validation loss decreased ({:.6f} --> {:.6f}).   Saving model ...'.format(local_loss,avg_train_loss))
                        torch.save({
                        'state_dict' : net.state_dict(),
                        'optimizer_state_dict' : optimizer.state_dict(),
                        'train_loss_min' : avg_train_loss
                        },'E:/NCKH2023/LPG-MI/checkpoints/target_model/target_ckp\VGG_MixedGhost.tar' )
                        local_loss =checkpoint['train_loss_min']= avg_train_loss

        return args.epochs, train

def main(epochs, trainf, testf, args):
    for epoch in range(epochs):
        trainf(epoch)
        testf(epoch)
def main_vgg(epochs, trainf, args):
    for epoch in range(epochs):
        trainf(epoch)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DP Training')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--eps', default=2, type=float, help='target epsilon')
    parser.add_argument('--grad_norm', '-gn', default=0.1,
                        type=float, help='max grad norm')
    parser.add_argument('--mode', default='ghost_mixed')
    parser.add_argument('--model', default='vgg16_bn', type=str)
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--cifar_data', type=str, default='celeba')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")


    if args.model == 'vgg16_bn':
        epochs, trainf= prepare(args)
        main_vgg(epochs,trainf,args)
    else:
        epochs, trainf, testf = prepare(args)
        main(epochs, trainf, testf, args)
