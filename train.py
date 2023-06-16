import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

from cmi import est_IMLY
import csv

def mi_reg(model, lin, x_train, y_train, device, args):
  I_MLY = est_IMLY(model, lin, x_train, y_train, device, round_=False, args=args)
  return torch.abs(I_MLY)

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None, model2=None, writer=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model.startswith('bert'):
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            if args.model.startswith('bert'):
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1] # [1] returns logits
            else:
                outputs = model(x)

            if args.cmi_reg and model2 is not None:
                #print('cmi-working')
                if args.ptft and epoch<args.pt_ep:
                    reg_val = 0.0
                else:
                    reg_val = mi_reg(model, model2, x, y, torch.device("cuda:0"), args)
            else:
                reg_val = 0.0
            if args.cmistinc:
               if args.ptft:
                   ep = epoch - args.pt_ep
               else:
                   ep = epoch
            else:
               ep = 0
               
            if args.gdro_alt and model2 is not None and is_training:
                if args.ptft and epoch<args.pt_ep:
                    g = g-g
                else:
                    m2o = torch.where(torch.gt(model2(torch.squeeze(x[:, :, 0].float())), torch.Tensor([0.0]).cuda()),
                                       torch.Tensor([1.0]).cuda(),
                                       torch.Tensor([0.0]).cuda())
                    g = m2o + 2*y
               
            if args.cmi_reg and writer is not None:
                writer.writerow({'CMI': reg_val})
            
            loss_main = loss_computer.loss(outputs, y, g, is_training, reg_val, ep=ep, scale=args.scale, th=args.th)

            if is_training:
                if args.model.startswith('bert'):
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()

def set_grad_false(model):
    for name, params in model.named_parameters():
        if name!='fc.weight' and name!='fc.bias':
            params.requires_grad_(False)
        else:
            print('not changing')
    return model

def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset, model2=None):
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight, 
        reg_st=args.reg_st)
    
    csvFile1 = open(args.log_dir+"/cmi_train.csv", "w")
    csvFile2 = open(args.log_dir+"/cmi_val.csv", "w")
    csvFile3 = open(args.log_dir+"/cmi_test.csv", "w")
    fieldnames = ['CMI']
    writer1 = csv.DictWriter(csvFile1, fieldnames=fieldnames)
    writer1.writeheader()
    writer2 = csv.DictWriter(csvFile2, fieldnames=fieldnames)
    writer2.writeheader()
    writer3 = csv.DictWriter(csvFile3, fieldnames=fieldnames)
    writer3.writeheader()

    # BERT uses its own scheduler and optimizer
    if args.model.startswith('bert'):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        print(f'\nt_total is {t_total}\n')
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=t_total)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        else:
            scheduler = None

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        #if args.cmi_reg and (model2 is None) and epoch==args.ep1:
        #    break
        
        if args.cmi_reg and args.ptft and epoch==args.pt_ep:
            model = set_grad_false(model)
      
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            model2=model2, writer=writer1)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha,
            reg_st = args.reg_st)
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False,
            model2 = model2, writer=writer2)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha,
                reg_st=args.reg_st)
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                None, test_csv_logger, args,
                is_training=False,
                model2 = model2, writer=writer3)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')


def test2(model, test_loader, args, groups=False, flag=False):
  model.eval()
  test_loss = 0
  correct = 0
  corr = 0
  #print(pl)
  with torch.no_grad():
   if args.model.startswith('bert'):
    for batch in test_loader:
      batch = tuple(t.cuda() for t in batch)
      data = batch[0]
      target = batch[1]
      gl = batch[2]
      #if args.model.startswith('bert'):
      #    input_ids = x[:, :, 0]
      #for data, target, gl in test_loader:
      data, target, gl = data.cuda(), target.cuda().float(), gl.cuda()
      if args.dataset=='MultiNLI':
          target=target.long()
      #gl = 1-((gl+1)//2)%2
      if groups:
         gl = 1-((gl+1)//2)%2
         target = gl.float()
      else:
         gl = gl-2*target
      #print(target)
      if flag:
        data=data[:,:,0]
     
      output = model(data.float())
      if args.dataset=='MultiNLI':
          test_loss += F.cross_entropy(torch.squeeze(output), target, reduction='sum').item()  # sum up batch loss
          pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
      else:
          test_loss += F.binary_cross_entropy_with_logits(torch.squeeze(output), target, reduction='sum').item()  # sum up batch loss
          pred = torch.where(torch.gt(output, torch.Tensor([0.0]).cuda()),
                         torch.Tensor([1.0]).cuda(),
                         torch.Tensor([0.0]).cuda())  # get the index of the max log-probability
      
      correct += pred.eq(target.view_as(pred)).sum().item()
      corr += pred.eq(gl.float().view_as(pred)).sum().item()
   else:   
    for data, target, gl in test_loader:
      data, target, gl = data.cuda(), target.cuda().float(), gl.cuda()
      #gl = 1-((gl+1)//2)%2
      if groups:
         gl = 1-((gl+1)//2)%2
         target = gl.float()
      else:
         gl = gl-2*target
      #print(target)
      
      output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(torch.squeeze(output), target, reduction='sum').item()  # sum up batch loss
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).cuda()),
                         torch.Tensor([1.0]).cuda(),
                         torch.Tensor([0.0]).cuda())  # get the index of the max log-probability
      
      correct += pred.eq(target.view_as(pred)).sum().item()
      corr += pred.eq(gl.float().view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Correlation: ({:.2f})\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset),
    corr / len(test_loader.dataset)))

  #return 100. * correct / len(test_loader.dataset)


def train2(model,criterion,data,args):
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=1e-4) #wtd=5e-4 for celeba
  epochs = args.ep1
  if args.model.startswith('bert'):
     bs = args.batch_size//2
  else:
     bs = args.batch_size//4
  dataloader = data['train_loader']
    
  model.cuda().train()
  for epoch in range(epochs):
    for i,datas in enumerate(dataloader):
     
     inputs,labels,gls=datas   
     inputs=inputs.cuda()
     if args.model.startswith('bert'):
       labels=labels.cuda().float()
     else:
       labels=labels.cuda()
     gls=gls.cuda()
     #gl = labels
     #gls = 1-((gls+1)//2)%2
     #print(labels)
     if args.groups:
        gls = 1-((gls+1)//2)%2
        labels = gls
     else:
        gls = gls - 2*labels
     #print(labels)
     optimizer.zero_grad()

     if args.model.startswith('bert'):
         if args.dataset=='MultiNLI':
           outputs = model(inputs.view(bs,128).float())
           labels = labels.long()
         else:
           outputs = model(inputs.view(bs,300).float())
         flag=True
     else:
         flag=False
         outputs = model(inputs.view(bs,3,224,224))
     
     if args.model.startswith('bert'):
       loss = criterion(outputs.squeeze().cuda(), (labels.view(bs,-1).squeeze()))
     else: 
       loss = criterion(outputs.squeeze().cuda(), (labels.view(bs,-1).squeeze()).float())
    
     loss.backward()
     optimizer.step()
     #if epoch%50==0:
     #  scheduler.step()
    if epoch%1==0:
      #scheduler.step()
      print('epoch {}, loss {}'.format(epoch, loss.item()))
      test2(model, data['train_loader'],args, args.groups)
      #if epoch%10==0:
      test2(model, data['test_loader'],args, args.groups,flag)
  for params in model.parameters():
    params.requires_grad_(False)
    
