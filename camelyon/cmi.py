import torch
import numpy as np

## Information Computations
# estimate I(M(x); y | L(x))

def kernel_val(x, mu, sig):
    a=torch.div(torch.exp(torch.div(-torch.square((x-mu)), 2*torch.square(torch.tensor(sig)))), torch.sqrt(torch.tensor(2*torch.pi)))
    return a

def getbinvals(M, cen_arr, Xv, Yv, device, sig=0.05):
    X=torch.zeros((4)).to(device)
    Y=torch.zeros((4)).to(device)
    for i in range(4):
    #for j in range(4):
        X[i]=kernel_val(Xv, cen_arr[i], sig)
        Y[i]=kernel_val(Yv, cen_arr[i], sig)
    M = torch.outer(torch.div(X, torch.sum(X)), torch.div(Y, torch.sum(Y)))
    return M
    
def est_IMYL(model, linModel, x, y, device, round_=True, bin_=False, args = None):
    Mx, Lx, y = est_pred(model, linModel, x, y)
    p = est_density(Mx, y, Lx, device, round_, bin_, args)
    return I_XYZ(p)

def est_IMLY(model, linModel, x, y, device, round_=True, bin_=False, args = None):
    Mx, Lx, y = est_pred(model, linModel, x, y, args)

    p = est_density(Mx, Lx, y, device, round_, bin_, args)
    return I_XYZ(p)

def est_pred(model, linModel, x, y, args=None):

    o1 = model(x)
    o2 = linModel(x)

    o1 = torch.nn.Softmax(dim=1)(o1)[:,1]
    o2 = torch.squeeze(torch.sigmoid(o2))

    Mx = torch.flatten(o1)
    Lx = torch.flatten(o2)
    y = torch.flatten(y)
    
    #p = est_density(Mx, y, Lx)
    return Mx, Lx, y
    
def est_density(X, Y, Z, device, round_=True, bin_=False, args = None): # estimate p[x,y,z] \in R^{{0,1}^3} for samples from X, Y, Z \in \N
    
    if not bin_: 
     if round_: 
      X = torch.round(X).int()
      Y = torch.round(Y).int()
     else:
      X = torch.sigmoid(12.5*(X-0.5))
      Y = torch.sigmoid(12.5*(Y-0.5))
    Z = torch.round(Z.float()).int()
    
    #if args is not None:
    #  n = torch.tensor([args.batch_size]).float().to(device)
    #else:
    n = X.size(dim=0)
    if bin_:
      p = torch.zeros((4,4,2)).to(device)
      cen_arr = torch.tensor([0.125, 0.375, 0.625, 0.875])
    else:
      p = torch.zeros((2, 2, 2)).to(device) # p[x,y,z] is the joint prob density
    
    flag=True
    for i in range(n):
      #if i == n-1: 
      #  print(i)
      #if i==1:
      #  flag=False
      if bin_:
        if round_:
         #i1 = (torch.round(2*X[i])+torch.round(X[i])).int()
         i1 = (1*(X[i]>0.25)+1*(X[i]>0.5)+1*(X[i]>0.75)).int()
         i2 = (1*(Y[i]>0.25)+1*(Y[i]>0.5)+1*(Y[i]>0.75)).int()
         #i2 = (torch.round(2*Y[i])+torch.round(Y[i])).int()
         p[i1, i2, Z[i]]+=1
        else:
         M = torch.zeros((4,4,2)).to(device)
         M = getbinvals(M, cen_arr, X[i], Y[i], device)
         #if flag:
         #  print(M)
         p[:,:,Z[i]] += M #/torch.sum(M, axis=(0,1))
      else:
        p[0, 0, Z[i]] += (1.0-X[i])*(1.0-Y[i])
        p[0, 1, Z[i]] += (1.0-X[i])*(Y[i])
        p[1, 0, Z[i]] += (X[i])*(1.0-Y[i])
        p[1, 1, Z[i]] += X[i]*Y[i]
    
    p /= n
    if bin_ and flag:
      print(p)
    return p

def est_density_mul(X, Y, Z, device, round_=True, bin_=False, args = None, nc=3): # estimate p[x,y,z] \in R^{{0,1}^3} for samples from X, Y, Z \in \N
    
    if not bin_: 
     if round_: 
      X = torch.round(X).int()
      Y = torch.round(Y).int()
     else:
      X = torch.sigmoid(12.5*(X-0.5))
      Y = torch.sigmoid(12.5*(Y-0.5))
    Z = torch.round(Z.float()).int()
    
    #if args is not None:
    #  n = torch.tensor([args.batch_size]).float().to(device)
    #else:
    n = X.size(dim=0)
    if bin_:
      p = torch.zeros((4,4,2)).to(device)
      cen_arr = torch.tensor([0.125, 0.375, 0.625, 0.875])
    else:
      p = torch.zeros((nc, nc, nc)).to(device) # p[x,y,z] is the joint prob density
    
    flag=True
    for i in range(n):
      #if i == n-1: 
      #  print(i)
      #if i==1:
      #  flag=False
      if bin_:
        if round_:
         #i1 = (torch.round(2*X[i])+torch.round(X[i])).int()
         i1 = (1*(X[i]>0.25)+1*(X[i]>0.5)+1*(X[i]>0.75)).int()
         i2 = (1*(Y[i]>0.25)+1*(Y[i]>0.5)+1*(Y[i]>0.75)).int()
         #i2 = (torch.round(2*Y[i])+torch.round(Y[i])).int()
         p[i1, i2, Z[i]]+=1
        else:
         M = torch.zeros((4,4,2)).to(device)
         M = getbinvals(M, cen_arr, X[i], Y[i], device)
         #if flag:
         #  print(M)
         p[:,:,Z[i]] += M #/torch.sum(M, axis=(0,1))
      else:
        for j in range(nc):
            for k in range(nc):
                p[j, k, Z[i]] += X[i,j]*Y[i,k]
    
    p /= n
    if bin_ and flag:
      print(p)
    return p

def I_XYZ(p): # compute I(X, Y | Z) for joint density p[x, y, z]
    pz = torch.sum(p, dim=(0,1), keepdim=True) # the density of z. pz[x,y,z] = p(z)
    
    p_xy_z = p / pz  # q[x, y, z] = p(x, y | z)
    p_x_z =  torch.sum(p, dim=1, keepdim=True) / pz  # p(x | z)
    p_y_z =  torch.sum(p, dim=0, keepdim=True) / pz  # p(y | z)
    
    I = torch.sum(p * torch.nan_to_num(torch.log2( p_xy_z / (p_x_z * p_y_z) )))
    return I

# returns I(A; B) where A, B \in {X, Y, Z} spefice by idx
# eg, I(X; Y) = I_ab(p, idx=[0, 1])
#     I(X; Z) = I_ab(p, idx=[0, 2])
def I_ab(p, idx=(0,1)): 
    exlude = (0+1+2) - np.sum(idx)
    p_ab = torch.sum(p, dim=exlude)
    p_a = torch.sum(p_ab, dim=1, keepdim=True)
    p_b = torch.sum(p_ab, dim=0, keepdim=True)
    
    I = torch.sum(p_ab * torch.nan_to_num(torch.log2( p_ab / (p_a * p_b) )))
    return I

def H(q): # binary entropy
    return -q*torch.log2(q) - (1-q)*torch.log2(1-q)



