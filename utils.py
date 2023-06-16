import sys
import os
import torch
import numpy as np
import csv
np.random.seed(10)

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class CSVBatchLogger:
    def __init__(self, csv_path, n_groups, mode='w'):
        columns = ['epoch', 'batch']
        for idx in range(n_groups):
            columns.append(f'avg_loss_group:{idx}')
            columns.append(f'exp_avg_loss_group:{idx}')
            columns.append(f'avg_acc_group:{idx}')
            columns.append(f'processed_data_count_group:{idx}')
            columns.append(f'update_data_count_group:{idx}')
            columns.append(f'update_batch_count_group:{idx}')
        columns.append('avg_actual_loss')
        columns.append('avg_per_sample_loss')
        columns.append('avg_acc')
        columns.append('model_norm_sq')
        columns.append('reg_loss')

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.columns = columns
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')

def get_balanced_data(train_data, dim=1):
  n = len(train_data)
  print('n=',n)
  #print(groups)
  #for i in range(100):
  #  print(train_data[i][1],train_data[i][2])
  
  idx = []
  for i in range(n):
     idx.append(train_data[i][dim])
  
  idx = np.asarray(idx)
  print(idx)
  
  n0=np.where(idx==0)[0]
  n1=np.where(idx==1)[0]

  if dim==2:
    n3=np.where(idx==3)[0]
    n2=np.where(idx==2)[0]
    #n0=min(n0,np.shape(n3)[0])
    #n1=min(n1,np.shape(n2)[0])
    print('n2 = ',len(n2),', n3 = ',len(n3))
    #print(max(n2),max(n3))
  
  print('n0 = ',len(n0),', n1 = ',len(n1))
  #print(max(n0),max(n1))

  n=min(np.shape(n0)[0],np.shape(n1)[0],20000)
  if dim==2:
    n=min(n,np.shape(n3)[0],np.shape(n2)[0])  

  a0=np.random.choice(n0,n,replace=False)
  a1=np.random.choice(n1,n,replace=False)
  a=np.concatenate((a0,a1))
  
  if dim==2:
    a3=np.random.choice(n3,n,replace=False)
    a2=np.random.choice(n2,n,replace=False)
    a=np.concatenate((a0,a1,a2,a3))
  
  new_data = []
  for i in a:
    new_data.append(train_data[i])
  return new_data


def get_balanced_data_gen(train_data, dim=1, args=None):
  n = len(train_data)
  print('n=',n)
  #print(groups)
  #for i in range(100):
  #  print(train_data[i][1],train_data[i][2])
  
  idx = []
  for i in range(n):
     idx.append(train_data[i][dim])
  
  idx = np.asarray(idx)
  print(idx)
  
  dict1={}
  num=[]

  for i in np.unique(idx):
    nn=np.where(idx==i)[0]
    dict1['n'+str(i)]=nn
    num.append(len(nn))

  nn=min(num)

  for i in np.unique(idx):
    if i==0:
      idx2=np.random.choice(dict1['n'+str(i)],nn,replace=False)
    else:
      idx2=np.concatenate((idx2,np.random.choice(dict1['n'+str(i)],nn,replace=False)))
  
  new_data = []
  for i in idx2:
    if args.model.startswith('bert'):
      new_data.append((train_data[i][0][:,0], train_data[i][1], train_data[i][2]))
    else:
      new_data.append(train_data[i])
  return new_data
