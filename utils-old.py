import torch
import random
import numpy as np
import os
from collections import Counter
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop


def set_seed(seed, use_gpu, print_out=True):
    if print_out:
        print('Seed:\t {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed(seed)


def update_correct_per_class(batch_output, batch_y, d):
    predicted_class = torch.argmax(batch_output, dim=-1)
    for true_label, predicted_label in zip(batch_y, predicted_class):
        if true_label == predicted_label:
            d[true_label.item()] += 1
        else:
            d[true_label.item()] += 0


def update_correct_per_class_topk(batch_output, batch_y, d, k):
    topk_labels_pred = torch.argsort(batch_output, axis=-1, descending=True)[:, :k]
    for true_label, predicted_labels in zip(batch_y, topk_labels_pred):
        d[true_label.item()] += torch.sum(true_label == predicted_labels).item()


def count_correct_topk(scores, labels, k):
    """Given a tensor of scores of size (n_batch, n_classes) and a tensor of
    labels of size n_batch, computes the number of correctly predicted exemples
    in the batch (in the top_k accuracy sense).
    """
    top_k_scores = torch.argsort(scores, axis=-1, descending=True)[:, :k]
    labels = labels.view(len(labels), 1)
    return torch.eq(labels, top_k_scores).sum()

def load_model(model, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    model.load_state_dict(d['model'])
    return d['epoch']

def load_optimizer(optimizer, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device)
    optimizer.load_state_dict(d['optimizer'])

def save(model, optimizer, epoch, location):
    dir = os.path.dirname(location)
    if not os.path.exists(dir):
        os.makedirs(dir)

    d = {'epoch': epoch,
         'model': model.state_dict(),
         'optimizer': optimizer.state_dict()}
    torch.save(d, location)


def decay_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    print('Switching lr to {}'.format(optimizer.param_groups[0]['lr']))
    return optimizer


def update_optimizer(optimizer, lr_schedule, epoch):
    if epoch in lr_schedule:
        optimizer = decay_lr(optimizer)
    return optimizer


class Plantnet(ImageFolder):
    def __init__(self, root, split, **kwargs):
        self.root = root
        self.split = split
        super().__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)


class MaxCenterCrop:
    def __call__(self, sample):
        min_size = min(sample.size[0], sample.size[1])
        return CenterCrop(min_size)(sample)


def get_data(size_image,root,batch_size, num_workers):

    transform = transforms.Compose(
        [MaxCenterCrop(), transforms.Resize(size_image), transforms.ToTensor()])

    trainset = Plantnet(root, 'images_train', transform=transform)
    train_class_to_num_instances = Counter(trainset.targets)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    testset = Plantnet(root, 'images_test', transform=transform)
    test_class_to_num_instances = Counter(testset.targets)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    n_classes = len(trainset.classes)

    dataset_attributes = {'n_train': len(trainset), 'n_test': len(testset), 'n_classes': n_classes,
                          'lr_schedule': [40, 50, 60],
                          'class2num_instances': {'train': train_class_to_num_instances,
                                                  'test': test_class_to_num_instances},
                          'class_to_idx': trainset.class_to_idx}

    return trainloader, testloader, dataset_attributes
