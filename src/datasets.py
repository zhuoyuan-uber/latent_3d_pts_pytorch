import numpy as np
import torch
import torch.utils.data as data

from .in_out import files_in_subdirs, load_point_clouds_from_filenames, pc_loader


class ShapenetCore(data.Dataset):
    def __init__(self, file_names, n_threads=8, verbose=False):
        self.file_names = file_names

        # pclouds is a numpy array of shape (batchsize, 2048, 3)
        pclouds, model_ids, syn_ids = load_point_clouds_from_filenames(file_names, n_threads, loader=pc_loader, verbose=verbose)
        pclouds = np.transpose(pclouds, (0, 2, 1))

        self.pclouds = torch.from_numpy(pclouds)

        #TODO: add noise if denoising autoencoder
        self.transform = None

    def __getitem__(self, index):
        data = self.pclouds[index]
        target = data.clone()

        if self.transform is not None:
            data = self.transform(data)

        return data, target

    def __len__(self):
        return len(self.pclouds)
