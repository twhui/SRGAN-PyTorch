from __future__ import print_function

def create_dataset(dataset_opt):
    if dataset_opt.mode == 'bicubic_down':
        from data.bicubic_down_dataset import BicubicDownDataset
        dataset = BicubicDownDataset()
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % dataset_opt.mode)
    dataset.initialize(dataset_opt)
    print('Dataset [%s] is created.' % dataset.name())
    return dataset
