import torch.utils.data

def create_dataloader(dataset, dataset_opt):
    phase = dataset_opt.phase
    if phase == 'train':
        batch_size = dataset_opt.batch_size
        shuffle = dataset_opt.use_shuffle
        num_workers = dataset_opt.n_workers
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=True) # Use paged-locked memory. Much Faster CPU <--> GPU data IO but comsume more physical memory.
