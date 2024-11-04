# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from mygpt import myGPT
from tokenizer import Tokenizer
from data import DataLoader, parse_lines, batchify,sft_parse_lines, sft_batchify
from optim import Optim

import argparse
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--ff_embed_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--train_data", type=str)
    parser.add_argument("--dev_data", type=str)
    parser.add_argument("--vocab", type=str)
    parser.add_argument("--min_occur_cnt", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--min_len", type=int)
    parser.add_argument("--print_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--start_from", type=str, default=None)
    parser.add_argument("--save_dir", type=str)

    parser.add_argument("--approx", type=str, default='none')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--MASTER_ADDR", type=str)
    parser.add_argument("--MASTER_PORT", type=str)
    parser.add_argument("--start_rank", type=int)
    parser.add_argument("--backend", type=str)
    parser.add_argument("--train_type", type=str,default="pretrain")

    return parser.parse_args()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def average_gradients(model):
    normal = True
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size
        else:
            normal = False
            break
    return normal


def eval_epoch(lm_args, model, tknizer, local_rank, label, batch_acm, train_type):
    ds = []
    with open(lm_args.dev_data, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ds.append(line)
    if train_type == "pretrain":
        ds = parse_lines(ds, lm_args.max_len, lm_args.min_len)
    else:
        ds = sft_parse_lines(ds, lm_args.max_len, lm_args.min_len)

    batch_size = 10
    batches = round(len(ds) / batch_size)
    idx = 0
    avg_nll = 0.
    avg_ppl = 0.
    avg_acc = 0.
    count_bsz = 0.
    count_tok = 0.
    while idx < len(ds):
        if train_type == "pretrain":
            ys_truth, ys_inp, msk = batchify(ds[idx: idx + batch_size], tknizer)
        else:
            ys_truth, ys_inp, msk = sft_batchify(ds[idx: idx + batch_size], tknizer)
        ys_truth = ys_truth.cuda(local_rank)
        ys_inp = ys_inp.cuda(local_rank)
        msk = msk.cuda(local_rank)
        acc, nll, ppl, toks, bsz = model.ppl(ys_truth, ys_inp, msk, lm_args.train_type)

        avg_acc += acc
        avg_nll += nll
        avg_ppl += ppl
        count_bsz += bsz
        count_tok += toks

        idx += batch_size

    print('valiadting: label %s, batch_acm %d, acc %.6f, nll %.6f, ppl %.6f' % (
        label, batch_acm, avg_acc/count_tok, avg_nll/count_bsz, avg_ppl/count_bsz), flush=True)


def run(args, local_rank):
    torch.manual_seed(1234)
    tknizer = Tokenizer(
        args.vocab, min_occur_cnt=args.min_occur_cnt, specials=[])
    if (args.world_size == 1 or dist.get_rank() == 0):
        print("vocab.size = %d" % tknizer.size, flush=True)
    model = myGPT(local_rank, tknizer, args.embed_dim,
                  args.ff_embed_dim, args.num_heads, args.dropout, args.layers)
    if args.start_from is not None:
        ckpt = torch.load(args.start_from, map_location='cpu')
        model.load_state_dict(ckpt['model'])
    model = model.cuda(local_rank)

    if args.world_size > 1:
        torch.manual_seed(1234 + dist.get_rank())
        random.seed(5678 + dist.get_rank())
    optimizer = Optim(model.embed_dim, args.lr, args.warmup_steps, torch.optim.AdamW(
        model.parameters(), lr=0, betas=(0.9, 0.998), eps=1e-9))

    if args.start_from is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    train_data = DataLoader(tknizer, args.train_data,
                            args.batch_size, args.max_len, args.min_len, args.train_type)
    batch_acm = 0
    acc_acm, nll_acm, ppl_acm, ntokens_acm, nxs, npairs_acm, loss_acm = 0., 0., 0., 0., 0., 0., 0.
    
    train_losses = []
    
    for epoch in tqdm(range(args.epoch), desc="Epoch Progress"):
        model.train()
        for truth, inp, msk in train_data:
            batch_acm += 1
            truth = truth.cuda(local_rank)
            inp = inp.cuda(local_rank)
            msk = msk.cuda(local_rank)
            model.zero_grad()
            res, loss, acc, nll, ppl, ntokens, npairs = model(truth, inp, msk, args.train_type)
            train_losses.append(loss.item())
            loss_acm += loss.item()
            acc_acm += acc
            nll_acm += nll
            ppl_acm += ppl
            ntokens_acm += ntokens
            npairs_acm += npairs
            nxs += npairs

            loss.backward()
            if args.world_size > 1:
                average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.print_every == -1 % args.print_every:
                print('batch_acm %d, loss %.3f, acc %.3f, nll %.3f, ppl %.3f, x_acm %d, lr %.6f' % (
                    batch_acm, loss_acm/args.print_every, acc_acm/ntokens_acm, nll_acm/nxs, ppl_acm/nxs, npairs_acm, optimizer._rate), flush=True)
                acc_acm, nll_acm, ppl_acm, ntokens_acm, loss_acm, nxs = 0., 0., 0., 0., 0., 0.

            if (args.world_size == 1 or dist.get_rank() == 0) and batch_acm % args.save_every == -1 % args.save_every:
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                torch.save({'args': args, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, '%s/epoch%d_batch_%d' % (args.save_dir, train_data.epoch_id, batch_acm))

                model.eval()
                eval_epoch(args, model, tknizer, local_rank, "epoch-" +
                           str(train_data.epoch_id) + "-acm-" + str(batch_acm), batch_acm, args.train_type)
                model.train()
                
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Batches')
    plt.legend()
    plt.grid()
    plt.savefig('train_loss.png')
    plt.show()


def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(
        backend, rank=args.start_rank + local_rank, world_size=args.world_size)
    fn(args, local_rank)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_config()
    if args.world_size == 1:
        run(args, 0)
        exit(0)
    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(
            args, rank, run, args.backend))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
