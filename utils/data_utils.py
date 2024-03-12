import logging

import torch
import torch.utils.data as data
import os
import json
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


logger = logging.getLogger(__name__)

## Can be modifed:
root_FFT = './data/FFT'
root_RFF = './data/RFF'

class FFT_DATASET(data.Dataset):
    def __init__(self, root, subset='train', block_size='512', scenario = 1, use_prior=False,transform=None, target_transform=None):

        super(FFT_DATASET, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.train = subset
        self.use_prior = use_prior
        self.target_change_rate = 0.00
        scenario=scenario
        # self.random_zero = 0.2

        self.data, self.targets, self.labels = self.load(scenario, block_size, subset)
        self.bit_shift_ratio = 0.0
        # self.data = self.data.astype(np.uint16)
        # self.data = self.data[:,::2] << 8 | self.data[:,1::2]
        # testing
        # self.data = self.data[:1000]
        # self.targets = self.targets[:1000]
        self.data = self.data.astype(np.int64)
        self.targets = self.targets.astype(np.int64)
        self.max_targets = self.targets.max() + 1
        print("Loaded {} data: data.shape={}, targets.shape={}".format(subset, self.data.shape, self.targets.shape))
        # data.shape=[n, 512], targets.shape=[n,]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        # if self.use_prior:
        #     if index != 0:
        #         if self.targets[index-1] != target and np.random.rand() < self.target_change_rate:
        #             data = np.append(data, np.random.randint(0,self.max_targets))
        #         else:
        #             data = np.append(data, self.targets[index-1])
        #     else:
        #         data = np.append(data, np.zeros((1,),dtype=np.int64))

        ## random bit shift
        if np.random.rand() < self.bit_shift_ratio:
            shift_bit = np.random.randint(1,8) #[1,7]
            data_a = data << shift_bit
            data_b = data >> (8-shift_bit)
            data_c = np.zeros_like(data_b)
            data_c[0:255] = data_b[1:256]
            data = data_a+data_c

        # ### data aug
        # if np.random.rand() < self.random_zero:
        #     zero_place = np.random.randint(0,256) #[0,255]
        #     zero_length = min(256, zero_place+np.random.randint(1,5))
        #     data[zero_place:zero_length] = 0

        data = data.astype(np.int64)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data)

    def getlabels(self):
        return self.labels

    def load(self, scenario=2, block_size='512', subset='train'):
        if block_size not in ['512', '4k']:
            raise ValueError('Invalid block size!')
        if scenario not in range(1, 7):
            raise ValueError('Invalid scenario!')
        if subset not in ['train', 'val', 'test']:
            raise ValueError('Invalid subset!')

        data_dir = os.path.join(self.root, '{:s}_{:1d}'.format(block_size, scenario))
        data = np.load(os.path.join(data_dir, '{}.npz'.format(subset)))

        if os.path.isfile(os.path.join(self.root,'classes.json')):
            with open(os.path.join(self.root,'classes.json')) as json_file:
                classes = json.load(json_file)
                labels = classes[str(scenario)]
        else:
            raise FileNotFoundError('Please download classes.json to the current directory!')

        return data['x'], data['y'], labels

class RFF_DATASET(data.Dataset):
    def __init__(self, root, subset='train', block_size='512', use_context=False, context_tokens=16,transform=None, target_transform=None):

        super(RFF_DATASET, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.train = subset
        self.context_tokens = context_tokens
        self.use_context = use_context
        self.target_change_rate = 0.05
        # self.temp_size = 100000
        # self.random_zero = 0.2

        self.data, self.targets, self.filename, self.labels = self.load(block_size, subset)
        self.bit_shift_ratio = 0.0
        # self.data = self.data.astype(np.uint16)
        # self.data = self.data[:,::2] << 8 | self.data[:,1::2]
        # testing
        # if subset=='val':
        #     self.data = self.data[:self.temp_size]
        #     self.targets = self.targets[:self.temp_size]
        self.data = self.data.astype(np.int64)
        self.targets = self.targets.astype(np.int64)
        self.max_targets = self.targets.max() + 1
        print("Loaded {} data: data.shape={}, targets.shape={}".format(subset, self.data.shape, self.targets.shape))
        # data.shape=[n, 512], targets.shape=[n,]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == 'train' and self.use_context:
            if index + self.context_tokens <= len(self.data):
                fetch_index_min = index
                fetch_index_max = index + self.context_tokens
                data, target = self.data[fetch_index_min:fetch_index_max], self.targets[fetch_index_min:fetch_index_max]
            else: #Tocheck
                data = np.concatenate((self.data[index:len(self.data)], self.data[0:self.context_tokens-(len(self.data)-index)]))
                target = np.concatenate((self.targets[index:len(self.data)], self.targets[0:self.context_tokens-(len(self.data)-index)]))
        else:
            data, target = self.data[index], self.targets[index]

        ## random bit shift
        if np.random.rand() < self.bit_shift_ratio:
            shift_bit = np.random.randint(1,8) #[1,7]
            data_a = data << shift_bit
            data_b = data >> (8-shift_bit)
            data_c = np.zeros_like(data_b)
            data_c[0:255] = data_b[1:256]
            data = data_a+data_c

        # ### data aug
        # if np.random.rand() < self.random_zero:
        #     zero_place = np.random.randint(0,256) #[0,255]
        #     zero_length = min(256, zero_place+np.random.randint(1,5))
        #     data[zero_place:zero_length] = 0

        data = data.astype(np.int64)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.data)

    def getlabels(self):
        return self.labels

    def getfilename(self, index):
        return self.filename[index]

    def load(self, block_size='512', subset='train'):
        if block_size not in ['512', '4k']:
            raise ValueError('Invalid block size!')
        if subset not in ['train', 'val', 'test']:
            raise ValueError('Invalid subset!')

        data_dir = os.path.join(self.root, '{:s}'.format(block_size))
        if subset=='train':
            data = np.load(os.path.join(data_dir, 'govdocs1_{}.npz'.format('train')))
            size = data['x'].shape[0]
            split_point  = int(size*4.0/5)
            data_x = data['x'][:split_point]
            data_y = data['y'][:split_point]
            data_z = data['z'][:split_point]
        elif subset=='val':
            data = np.load(os.path.join(data_dir, 'govdocs1_{}.npz'.format('train')))
            size = data['x'].shape[0]
            split_point  = int(size*4.0/5)
            data_x = data['x'][split_point:]
            data_y = data['y'][split_point:]
            data_z = data['z'][split_point:]
        else:
            data = np.load(os.path.join(data_dir, 'govdocs1_{}.npz'.format(subset)))
            data_x, data_y, data_z = data['x'], data['y'], data['z']

        # labels = ['jpg', 'gif', 'doc', 'xls', 'ppt', 'html', 'text', 'pdf']
        labels = ['jpg', 'gif', 'doc', 'xls', 'ppt', 'html', 'text', 'pdf',
                  'rtf', 'png', 'log', 'csv', 'gz', 'swf', 'eps', 'ps']
        # labels = ['jpg', 'gif', 'doc', 'xls', 'ppt', 'html', 'text', 'pdf', 'gz']
        # labels = ['jpg', 'gif', 'doc', 'xls', 'html', 'text', 'pdf', 'gz']

        return data_x, data_y, data_z, labels

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if "fft" in args.dataset:
        #/media/wangyi/System/data/FFT
        #/home/wangyi/data/FFT
        block_size = '512' if args.emb_num == 512 else '4k' if args.emb_num == 4096 else None
        trainset = FFT_DATASET(root=root_FFT, use_prior=args.use_prior,
                               subset='train', block_size=block_size, scenario=int(args.dataset[-1]))
        testset = FFT_DATASET(root=root_FFT, use_prior=args.use_prior,
                               subset='test', block_size=block_size, scenario=int(args.dataset[-1]))
        labels = testset.getlabels()
    elif "rff" in args.dataset:
        block_size = '512' if args.emb_num == 512 else '4k' if args.emb_num == 4096 else None
        trainset = RFF_DATASET(root=root_RFF, use_context = args.use_context,
                               subset='train', block_size=block_size)
        testset = RFF_DATASET(root=root_RFF,
                              subset='val', block_size=block_size)
        labels = testset.getlabels()
    else:
        raise ValueError('Invalid dataset name!')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              drop_last=True if args.use_context else False,
                              pin_memory=False)
    if args.use_prior:
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=1,
                                 num_workers=args.num_workers,
                                 pin_memory=False) if testset is not None else None
    else:
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers,
                                 # drop_last=True if args.use_context else False,
                                 pin_memory=False) if testset is not None else None

    return train_loader, test_loader, labels

def get_loader_test(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if "fft" in args.dataset and "dfrws" not in args.dataset and "govdocs" not in args.dataset:
        #/media/wangyi/System/data/FFT
        #/home/wangyi/data/FFT
        block_size = '512' if args.emb_num == 512 else '4k' if args.emb_num == 4096 else None
        testset = FFT_DATASET(root=root_FFT, use_prior=args.use_prior,
                              subset='test', block_size=block_size, scenario=int(args.dataset[-1]))
        labels = testset.getlabels()
    elif "rff" in args.dataset:
        block_size = '512' if args.emb_num == 512 else '4k' if args.emb_num == 4096 else None
        testset = RFF_DATASET(root=root_RFF,
                              subset='test', block_size=block_size)
        labels = testset.getlabels()
    else:
        raise ValueError('Invalid dataset name!')

    if args.local_rank == 0:
        torch.distributed.barrier()

    test_sampler = SequentialSampler(testset)
    if args.use_prior:
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=1,
                                 num_workers=args.num_workers,
                                 pin_memory=True) if testset is not None else None
    else:
        test_loader = DataLoader(testset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers,
                                 # drop_last=True if args.use_context else False,
                                 pin_memory=True) if testset is not None else None

    return test_loader, labels
