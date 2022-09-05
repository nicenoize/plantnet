import os
import csv
from tqdm import tqdm
import time
import torch
from torch import nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchvision.models import vit_b_16
from torchvision.models import resnet50 
from torchvision.models import resnet18
from torchvision.models import resnext101_32x8d
from utils import set_seed, update_optimizer, get_data, save, load_model
from epoch import train_epoch, test_epoch


def train(mu,lr,batch_size,n_epochs,k,model,use_gpu,size_image,seed,num_workers,root):
    set_seed(seed, use_gpu)
    train_loader, test_loader, dataset_attributes = get_data(size_image,root,batch_size, num_workers)
  
    for param in model.parameters():
        param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default

    if model.__class__.__name__ == 'ResNet':
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, dataset_attributes['n_classes'])
    elif model.__class__.__name__ == 'VisionTransformer':
        input_features = model.heads[0].in_features
        model.heads[0] = nn.Linear(input_features, dataset_attributes['n_classes'])
    elif model.__class__.__name__ == 'Inception3':
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, dataset_attributes['n_classes'])
    
    criteria = CrossEntropyLoss()

    if use_gpu:
        torch.cuda.set_device(0)
        model.cuda()
        criteria.cuda()

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=mu, nesterov=True)
    
    
    # Containers for storing metrics over epochs
    loss_train, acc_train, f1_train, topk_acc_train = [], [], [], []
    loss_test, acc_test, f1_test, topk_acc_test = [], [], [], []
    
    save_dir = '/plantnet'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('k : ', k)

    best_acc = 0.0

    for epoch in tqdm(range(n_epochs), desc='epoch', position=0):
        t = time.time()
        optimizer = update_optimizer(optimizer, lr_schedule=dataset_attributes['lr_schedule'], epoch=epoch)

        loss_epoch_train, f1_epoch_train, acc_epoch_train, topk_acc_epoch_train = train_epoch(model, optimizer, train_loader,
                                                                              criteria, loss_train, f1_train, acc_train,
                                                                              topk_acc_train, k,
                                                                              dataset_attributes['n_train'],
                                                                              use_gpu)
        

    #load_model(model, os.path.join(save_dir, 'weights_best_acc.tar'), use_gpu)
   
        loss_epoch_test, f1_epoch_test, acc_epoch_test, topk_acc_epoch_test, \
        class_acc_test = test_epoch(model, test_loader, criteria, 
                                    loss_test, acc_test, f1_test, topk_acc_test,
                                    k, use_gpu, dataset_attributes)

        """ print()
        print(f'epoch {epoch} took {time.time()-t:.2f}')
        print(f'loss_train : {loss_epoch_train}')
        print(f'f1_train : {f1_epoch_train}')
        print(f'acc_train : {acc_epoch_train} / topk_acc_train : {topk_acc_epoch_train}')
        print(f'loss_test : {loss_epoch_test}')
        print(f'f1_test : {f1_epoch_test}')
        print(f'acc_test : {acc_epoch_test} / topk_acc_train : {topk_acc_epoch_test}') """

        if acc_epoch_test > best_acc:
            best_acc = acc_epoch_test
            save(model, optimizer, epoch, os.path.join(save_dir, 'weights_best_acc.tar'))

    """ results = {'loss_train': loss_train, 'f1_train': f1_train, 'acc_train': acc_train, 'topk_acc_train': topk_acc_train, 
            'test_results': {'loss_test': loss_test,
                                'f1_test': f1_test,
                                'acc_test': acc_test,
                                'topk_acc_test': topk_acc_test,
                                'class_acc_dict': class_acc_test}
            } """
    
              
    #Writing results to csv file
    with open(save_dir+'/results_%s_%s.csv' %(model.name,n_epochs), 'w', newline='') as csvfile:
        fieldnames = ['loss_train','acc_train', 'topk_acc_train', 'f1_train','loss_test','acc_test', 'topk_acc_test', 'f1_test']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(loss_train)):
            writer.writerow({'loss_train':loss_train[i],'acc_train':acc_train[i],'topk_acc_train':topk_acc_train[i].get(k[0]),'f1_train':f1_train[i],'loss_test':loss_test[i],'acc_test':acc_test[i],'topk_acc_test':topk_acc_test[i].get(k[0]),'f1_test':f1_test[i]})

if __name__ == '__main__':
    #Loss parameters
    mu = 0.0001 #weight decay parameter

    #Training parameters
    lr = 0.01 #learning rate to use RESNET: 0.01  VIT: 0.0005
    batch_size = 32   #For all models: 32
    n_epochs = 30  #RESNET: 30 VIT: 20
    k = [5] #top-k-evaluation

    #Model parameters
    #choose the model you want to train on
    # Vision Transformer
    #----------------------------------------------------------
    #model = vit_b_16(pretrained=True)
    #model.name = 'VisionTransformer'

    #ResNext
    #----------------------------------------------------------
    #model = resnext101_32x8d(pretrained=True)
    #model.name = 'Resnext101'

    #ResNet
    #----------------------------------------------------------
    model = resnet18(pretrained=True)
    model.name = 'Resnet18'
    
    #Hardware parameters
    use_gpu = torch.cuda.is_available()

    #Image Size
    if model.__class__.__name__ == 'ResNet':
        size_image = 224
    elif model.__class__.__name__ == 'VisionTransformer':
        size_image = 224
    else:
        print("Model not supported") 
        quit()

    # Miscellaneous parameters
    seed= 0 # set the seed for reproductible experiments
    num_workers=4 # increase this value to use multiprocess data loading. Default is one. You can bring it up. If you have memory errors go back to one
    root='/plantnet' #location of the train val and test directories


    train(mu,lr,batch_size,n_epochs,k,model,use_gpu,size_image,seed,32,root)
