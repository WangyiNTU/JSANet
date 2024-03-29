# coding=utf-8
from __future__ import absolute_import, division, print_function

import warnings
warnings.filterwarnings("ignore")

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from models.modeling2 import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader_test
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, accuracy):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_best.pth" % (args.name))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_model_last(args, model, accuracy):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % (args.name))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.emb_num = args.emb_num
    config.use_ca = args.use_ca
    config.use_cnn12 = args.use_cnn12
    args.use_prior = True if config.classifier == 'prior' else False
    args.use_context = config.use_context = True if config.classifier == 'context' else False
    if args.use_context:
        config.context_tokens = args.context_tokens

    if "fft1" in args.dataset:
        num_classes = 75
    elif "fft2" in args.dataset:
        num_classes = 11
    elif "fft3" in args.dataset:
        num_classes = 25
    elif "fft4" in args.dataset:
        num_classes = 5
    elif "fft5" in args.dataset:
        num_classes = 2
    elif "fft6" in args.dataset:
        num_classes = 2
    elif "rff" in args.dataset or "fft7" in args.dataset: # fft7=rff when using for govdocs and dfrws testing
        num_classes = 16
    else:
        print("No such dataset")
        exit(0)

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    if args.pretrained_dir is not None:
        model.load_state_dict(torch.load(args.pretrained_dir, map_location='cuda:0'))
        print('Load model from {}'.format(args.pretrained_dir))
    if len(args.gpus) > 0:
        model.to(args.device)
        model = torch.nn.DataParallel(model, args.gpus)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.4fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, test_loader, labels=None):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, y = batch
        pad_x_flag = False
        if args.use_context and x.shape[0] < args.eval_batch_size:
            ori_batch_size = x.shape[0]
            pad_x = torch.zeros([args.eval_batch_size-x.shape[0],x.shape[1]],dtype=torch.int64)
            x = torch.cat([x, pad_x], dim=0)
            batch = (x, y)
            pad_x_flag = True

        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():
            logits = model(x)[0]
            if pad_x_flag:
                logits = logits[:ori_batch_size,:]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.avg)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    correct_list = (all_preds==all_label).astype(np.single)
    correct_meter_subset = {}
    wrong_meter_subset = {}
    for label_i, label_name in enumerate(labels):
        if correct_list[all_label==label_i].shape[0] == 0:
            continue
        correct_subset = correct_list[all_label==label_i]
        num_correct = correct_subset.sum()
        num_subset = correct_subset.shape[0]
        correct_meter_subset[label_name] = num_correct/num_subset
        # mis classification
        all_preds_lable_i = all_preds[all_label==label_i]
        unique, counts = np.unique(all_preds_lable_i, return_counts=True)
        all_preds_lable_dict = dict(zip(unique, counts))
        all_preds_lable_dict.pop(label_i, None)
        if not all_preds_lable_dict:
            wrong_meter_subset[label_name] = ['None', 0.0]
        else:
            max_mis_class = max(all_preds_lable_dict, key=all_preds_lable_dict.get)
            wrong_meter_subset[label_name] = [labels[max_mis_class], all_preds_lable_dict[max_mis_class]/num_subset]

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    avg_acc = AverageMeter()
    for subset_key, subset_value in correct_meter_subset.items():
        misclass = wrong_meter_subset[subset_key][0]
        misclass_prob = wrong_meter_subset[subset_key][1]
        logger.info('Valid Accuracy of subset {} is {:.4f} with max misclass {} of {:.4f}'.
                    format(subset_key, subset_value, misclass, misclass_prob))
        avg_acc.update(subset_value, 1)
    logger.info("Average Accuracy of Diagonal: %2.5f" % avg_acc.avg)

    return accuracy


def test(args, model):

    # Prepare dataset
    test_loader, labels = get_loader_test(args)

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running testing *****")
    accuracy = valid(args, model, test_loader, labels)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="fft2",
                        help="Which downstream task.")
    parser.add_argument("--emb_num", default=512, type=int,
                        help="embedding size (memory block size)")
    parser.add_argument("--model_type", choices=["ViT-S_8","ViT-S_4","ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16", "CNN", "ViT-S_1",
                                                 "ViT-S_1_P", "ViT-S_1C", "Long"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--use_ca", action='store_true',
                        help="use channel attention")
    parser.add_argument("--use_cnn12", action='store_true',
                    help="use cnn12, lower priority than use_ca")
    parser.add_argument("--pretrained_dir", type=str, default=None, #"checkpoint/ViT-B_16.npz"
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--context_tokens', default=16, type=int)

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1, type=int,
                        help="Run prediction on validation set every so many epochs."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--gpus', type=str, help='gpu_id',default='0')
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        str_ids = args.gpus.split(',')
        args.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpus.append(id)
        if len(args.gpus) > 0:
            torch.cuda.set_device(args.gpus[0])
        device = args.gpus[0]
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    test(args, model)


if __name__ == "__main__":
    main()
