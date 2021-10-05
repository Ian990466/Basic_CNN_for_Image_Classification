import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
from network.prosecutor import *

def model_training(model, n_epochs, train_loader, valid_loader, optimizer, criterion):
    train_acc_his, valid_acc_his= [],[]
    train_losses_his, valid_losses_his= [],[]
    # train_on_gpu = True
    for epoch in range(n_epochs):
        # keep track of training and validation loss
        train_loss, valid_loss = 0.0, 0.0
        train_losses, valid_losses= [],[]
        train_correct, val_correct,train_total, val_total= 0,0,0,0
        train_pred, train_target= torch.zeros(8,1), torch.zeros(8,1)
        val_pred, val_target= torch.zeros(8,1), torch.zeros(8,1)
        count= 0
        count2= 0
        print('epoch: {}'.format(epoch + 1))
        
        # train the model
        # model.cuda()
        model.train()
        for (data, target) in tqdm(train_loader):
            # print(target)
            # move tensors to GPU if CUDA is available
            # if train_on_gpu:
            #     data, target= data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            #calculate accuracy
            pred= output.data.max(dim= 1, keepdim= True)[1]
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item() * data.size(0))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if count == 0:
                train_pred=pred
                train_target=target.data.view_as(pred)
                count= count+1
            else:
                train_pred= torch.cat((train_pred,pred), 0)
                train_target= torch.cat((train_target,target.data.view_as(pred)), 0)
        train_pred=train_pred.cpu().view(-1).numpy().tolist()
        train_target=train_target.cpu().view(-1).numpy().tolist()

        # validate the model
        model.eval()
        for (data, target) in tqdm(valid_loader):
            # move tensors to GPU if CUDA is available
            # if train_on_gpu:
            #     data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss =criterion(output, target)
            #calculate accuracy
            pred = output.data.max(dim = 1, keepdim = True)[1]
            val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            val_total += data.size(0)
            valid_losses.append(loss.item()*data.size(0))
            if count2==0:
                val_pred=pred
                val_target=target.data.view_as(pred)
                count2=count+1
            else:
                val_pred = torch.cat((val_pred,pred), 0)
                val_target = torch.cat((val_target,target.data.view_as(pred)), 0)
        val_pred=val_pred.cpu().view(-1).numpy().tolist()
        val_target=val_target.cpu().view(-1).numpy().tolist()
            
        # calculate average losses
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
            
        # calculate average accuracy
        train_acc = train_correct/train_total
        valid_acc = val_correct/val_total
        train_acc_his.append(train_acc)
        valid_acc_his.append(valid_acc)
        train_losses_his.append(train_loss)
        valid_losses_his.append(valid_loss)
        # print training/validation statistics 
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))
        print('\tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            train_acc, valid_acc))

    return train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model

def result_graph(train_losses_his, valid_losses_his, train_acc_his, valid_acc_his):
    # plt accuracy and loss
    plt.figure(figsize=(15,10))
    plt.subplot(221)
    plt.plot(train_losses_his, 'bo', label = 'training loss')
    plt.plot(valid_losses_his, 'r', label = 'validation loss')
    plt.title("ResNet Loss")
    plt.legend(loc='upper left')
    plt.subplot(222)
    plt.plot(train_acc_his, 'bo', label = 'trainingaccuracy')
    plt.plot(valid_acc_his, 'r', label = 'validation accuracy')
    plt.title("ResNet Accuracy")
    plt.legend(loc = 'upper left')
    plt.show()

def main():
    args = parse.parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    train_path = args.train_path
    output_path = args.output_path

    # Data transforms setting    
    data_transforms = {
        'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    }

    output_path = output_path + "/" + model_name + "/"
    # Folder exist or not
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # cuDNN nn model optimzation
    # torch.backends.cudnn.benchmark= True

    # Load training dataset
    train_dataset = torchvision.datasets.ImageFolder(train_path, transform = data_transforms['train'])
    # print(train_dataset)
    # print(train_dataset.class_to_idx)

    # Split training dataset
    train_size = int(0.7 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
    # print("Num of Classes: " +  str(len(.classes)))
    print("Training Size: " + str(train_size))
    print("Training Size: " + str(valid_size))

    # Creat dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle= True, drop_last= False, num_workers= 8)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size= batch_size, shuffle= True, drop_last= False, num_workers= 8)

    print(train_loader)

    n_epochs= 1
    model = Basic_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-08)
    train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model = model_training(model, n_epochs, train_loader, val_loader, optimizer, criterion)
    torch.save(model, output_path + "Deepfake_Model_with_CNN.pt")

    result_graph(train_acc_his, valid_acc_his, train_losses_his, valid_losses_his)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--model_name', '-n', type= str, default= "Deepfake_Model_with_CNN")
    parse.add_argument('--batch_size', '-bs', type= int, default= 8)
    parse.add_argument('--train_path', '-tp', type= str, default= './datasets/')
    parse.add_argument('--output_path', '-mp', type= str, default= './output/')
    main()