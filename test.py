import numpy as np
import random
import torch
import torch.optim as optim

from chamfer_distance import ChamferDistance


chamfer_dist = ChamferDistance()

def my_loss(loss, pc1, pc2, print_flag=False):
    dist1, dist2 = loss(pc1, pc2)
    l = (torch.sum(dist1)) + (torch.sum(dist2))
    if print_flag:
        # print('dist1', dist1)
        # print('dist2', dist2)
        print('loss', l)
    return dist1, dist2, l


if __name__ == "__main__":
    #...
    # points and points_reconstructed are n_points x 3 matrices
    random.seed(100)
    np.random.seed(100)

    xyz1 = np.random.randn(32, 16384, 3).astype('float32')
    xyz2 = np.random.randn(32, 1024, 3).astype('float32')
    # confirm 1: random points are the same (done)
    # print('xyz1', xyz1)
    # print('xyz2', xyz2)

    # confirm 2: distance are the same (cpu)
    cd = ChamferDistance()
    pc1 = torch.from_numpy(xyz1)
    pc2 = torch.from_numpy(xyz2)
    # dist1, dist2 = cd(pc1, pc2)
    # loss = (torch.sum(dist1)) + (torch.sum(dist2))
    dist1, dist2, l = my_loss(cd, pc1, pc2, True)
    print('cpu done.')
    pc1 = pc1.cuda()
    pc2 = pc2.cuda()
    dist1, dist2, l = my_loss(cd, pc1, pc2, True)
    print('gpu done.')
    # print(dist1)
    # print(dist2)

    # confirm 3: backward
    pc1.requires_grad = True
    pc2.requires_grad = True
    optimizer = optim.SGD([pc1, pc2], lr=.01) # , momentum=.9)

    for epoch in range(100):
        optimizer.zero_grad()
        print(epoch, )
        dist1, dist2, l = my_loss(cd, pc1, pc2, True)
        l.backward()
        optimizer.step()
