import os
import logging

import random
import numpy as np

import torch
import torch.nn.functional as F

checkpoint_dir = os.getcwd() + '/models'
cuda_is_available = torch.cuda.is_available()

def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(torch.autograd.Variable(x.cuda(async=True), volatile=volatile))

def cuda(x):
    return x.cuda() if cuda_is_available else x

def train_and_validate(
    train_data_loader,
    valid_data_loader,
    model,
    optimizer,
    scheduler,
    looker,
    loss_fn,
    epochs,
    start_epoch,
    best_val_loss,
    alpha,
    center_loss_weight,
    experiment_name,
):
    if best_val_loss is None:
        best_val_loss = float('+inf')
    
    valid_losses = []
    lr_reset_epoch = start_epoch
    
    for epoch in range(start_epoch, epochs+1):
        loss, acc = train(
            train_data_loader,
            model,
            optimizer,
            loss_fn,
            epoch,
            alpha,
            center_loss_weight,
        )
        val_loss,val_acc = validate(
            valid_data_loader,
            model,
            loss_fn,
            alpha,
            center_loss_weight,
        )
        valid_losses.append(loss)
        
        if loss < best_val_loss:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                    'centers': model.centers,
                },
                '{experiment_name}_{epoch}_{loss}_{acc}_{val_acc}.pth'.format(experiment_name=experiment_name, epoch=epoch, loss=loss,acc=acc,val_acc=val_acc),
                checkpoint_dir,
            )
            best_val_loss = loss
        scheduler.step(loss, epoch)
        if looker.step(loss, epoch):
            break
    return model

def train(train_loader, model, optimizer, criterion, epoch,alpha,center_loss_weight):
    losses = []
    accuracy_scores = []
    
    model.train()
    
    logging.info('Epoch: {}'.format(epoch))
    print('Epoch: {}'.format(epoch))
    for i, (inputs, O, targets) in enumerate(train_loader):
        inputs, O, targets = variable(inputs.cuda()), variable(O.cuda()), variable(targets)
        outputs = model.forward(inputs, O)
        cross_loss = criterion(outputs, targets)
        loss = cross_loss 
        #loss = cross_loss 
        optimizer.zero_grad()
        batch_size = inputs.size(0)
        (batch_size * loss).backward()
        optimizer.step()
        losses.append(cross_loss.data[0])
        accuracy_scores += list(targets.data.cpu().numpy() == np.argmax(outputs.data.cpu().numpy(), axis=1))
        
        if i % 50 == 0:
            print('Step: {}, train_loss: {}'.format(i, np.mean(losses[-50:])))
        #if i % 100 == 0:
        #    logging.info('Step: {}, train_loss: {}'.format(i, np.mean(losses[-100:])))
            
    train_loss = np.mean(losses)
    train_accuracy = np.mean(accuracy_scores)
    print('train_loss: {}, train_acc: {}'.format(train_loss,train_accuracy))
    logging.info('train_loss: {}, train_acc: {}'.format(train_loss, train_accuracy))
    return train_loss, train_accuracy

def validate(val_loader, model, criterion,alpha,center_loss_weight):
    #print('in valid')
    accuracy_scores = []
    losses = []
    
    model.eval()
    
    for i, (inputs, O, targets) in enumerate(val_loader):
        inputs, O, targets = variable(inputs.cuda(), volatile=True), variable(O.cuda()), variable(targets)
        outputs = model.forward(inputs, O)
        cross_loss = criterion(outputs, targets)
        loss = cross_loss 
        
        losses.append(loss.data[0])
        
        accuracy_scores += list(targets.data.cpu().numpy() == np.argmax(outputs.data.cpu().numpy(), axis=1))
        
    valid_loss, valid_accuracy = np.mean(losses), np.mean(accuracy_scores)
    logging.info('valid_loss: {}, valid_acc: {}'.format(valid_loss, valid_accuracy))
    print('valid_loss: {}, valid_acc: {}'.format(valid_loss, valid_accuracy))
    return valid_loss,valid_accuracy

def save_checkpoint(state, filename, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
