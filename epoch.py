import torch
from torchmetrics.functional import f1_score
from tqdm import tqdm
from utils import count_correct_topk, update_correct_per_class, \
    update_correct_per_class_topk
import torch.nn.functional as F
from collections import defaultdict
import torchvision.transforms as transforms




def train_epoch(model, optimizer, train_loader, criteria, loss_train, f1_train, acc_train, topk_acc_train, list_k, n_train, use_gpu):
    """Single train epoch pass. At the end of the epoch, updates the lists loss_train, acc_train and topk_acc_train"""
    model.train()
    # Initialize variables
    f1_epoch_train = 0
    loss_epoch_train = 0
    n_correct_train = 0
    # Containers for tracking nb of correctly classified examples (in the top-k sense) and top-k accuracy for each k in list_k
    n_correct_topk_train = defaultdict(int)
    topk_acc_epoch_train = {}

    
    for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_loader, desc='train', position=0)):
        #transform = transforms.Compose([
        #    #transforms.AutoAugmentPolicy(transforms.AutoAugmentPolicy.IMAGENET),
        #    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #    transforms.ToTensor()
        #])
        #batch_x_train = transform(batch_x_train)
        #batch_y_train = transform(batch_y_train)
        if use_gpu:
            batch_x_train, batch_y_train = batch_x_train.cuda(), batch_y_train.cuda()
        optimizer.zero_grad()
        

        batch_output_train = model(batch_x_train)
        loss_batch_train = criteria(batch_output_train, batch_y_train)
        loss_epoch_train += loss_batch_train.item()
        loss_batch_train.backward()
        optimizer.step()
        f1_epoch_train += f1_score(torch.argmax(batch_output_train, dim=-1), batch_y_train).item()

        # Update variables
        with torch.no_grad():
            n_correct_train += torch.sum(torch.eq(batch_y_train, torch.argmax(batch_output_train, dim=-1))).item()
            for k in list_k:
                n_correct_topk_train[k] += count_correct_topk(scores=batch_output_train, labels=batch_y_train, k=k).item()
    

    # At the end of epoch compute average of statistics over batches and store them
    with torch.no_grad():
        loss_epoch_train /= batch_idx
        f1_epoch_train /= batch_idx
        epoch_accuracy_train = n_correct_train / n_train
        for k in list_k:
            topk_acc_epoch_train[k] = n_correct_topk_train[k] / n_train

        loss_train.append(loss_epoch_train)
        f1_train.append(f1_epoch_train)
        acc_train.append(epoch_accuracy_train)
        topk_acc_train.append(topk_acc_epoch_train)

    return loss_epoch_train, f1_epoch_train, epoch_accuracy_train, topk_acc_epoch_train

def test_epoch(model, test_loader, criteria, loss_test, acc_test, f1_test, topk_acc_test, list_k, use_gpu, dataset_attributes): 

    print()
    model.eval()
    with torch.no_grad():
        n_test = dataset_attributes['n_test']
        loss_epoch_test = 0
        f1_epoch_test = 0
        n_correct_test = 0
        topk_acc_epoch_test = {}
        n_correct_topk_test = defaultdict(int)

        class_acc_dict = {}
        class_acc_dict['class_acc'] = defaultdict(int)
        class_acc_dict['class_topk_acc'] = {}
        for k in list_k:
            class_acc_dict['class_topk_acc'][k] = defaultdict(int)

        for batch_idx, (batch_x_test, batch_y_test) in enumerate(tqdm(test_loader, desc='test', position=0)):
            if use_gpu:
                batch_x_test, batch_y_test = batch_x_test.cuda(), batch_y_test.cuda()
            
            batch_output_test = model(batch_x_test)
            batch_proba_test = F.softmax(batch_output_test)
            loss_batch_test = criteria(batch_output_test, batch_y_test)
            loss_epoch_test += loss_batch_test.item()

            f1_epoch_test += f1_score(torch.argmax(batch_output_test, dim=-1), batch_y_test).item()

            n_correct_test += torch.sum(torch.eq(batch_y_test, torch.argmax(batch_output_test, dim=-1))).item()
            update_correct_per_class(batch_proba_test, batch_y_test, class_acc_dict['class_acc'])
            for k in list_k:
                n_correct_topk_test[k] += count_correct_topk(scores=batch_output_test, labels=batch_y_test, k=k).item()
                update_correct_per_class_topk(batch_output_test, batch_y_test, class_acc_dict['class_topk_acc'][k], k)

        # After seeing test set update the statistics over batches and store them
        loss_epoch_test /= batch_idx
        f1_epoch_test /= batch_idx
        acc_epoch_test = n_correct_test / n_test
        for k in list_k:
            topk_acc_epoch_test[k] = n_correct_topk_test[k] / n_test

        for class_id in class_acc_dict['class_acc'].keys():
            n_class_test = dataset_attributes['class2num_instances']['test'][class_id]
            class_acc_dict['class_acc'][class_id] = class_acc_dict['class_acc'][class_id] / n_class_test
            for k in list_k:
                class_acc_dict['class_topk_acc'][k][class_id] = class_acc_dict['class_topk_acc'][k][class_id] / n_class_test
        
        loss_test.append(loss_epoch_test)
        f1_test.append(f1_epoch_test)
        acc_test.append(acc_epoch_test)
        topk_acc_test.append(topk_acc_epoch_test)

    return loss_epoch_test, f1_epoch_test, acc_epoch_test, topk_acc_epoch_test, class_acc_dict
