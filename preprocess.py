# -*- coding: utf-8 -*-
import os
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-src', default=None, help='Trainig source file')
    parser.add_argument('--train-tgt', default=None, help='Training target file')
    parser.add_argument('--valid-src', default=None, help='Validation source file')
    parser.add_argument('--valid-tgt', default=None, help='Validation target file')
    parser.add_argument('--maxlen', type=int, default=70, help='Max number of words')
    parser.add_argument('--out-train', default=None, help='Output train TSV file (.train.tsv)')
    parser.add_argument('--out-valid', default=None, help='Output valid TSV file (.valid.tsv)')
    args = parser.parse_args()

    if args.train_src and args.train_tgt and args.out_train:
        skip_train = False
    elif not args.train_src and not args.train_tgt and not args.out_train:
        skip_train = True
    else:
        raise "Missing options"

    if args.valid_src and args.valid_tgt and args.out_valid:
        skip_valid = False
    elif not args.valid_src and not args.valid_tgt and not args.out_valid:
        skip_valid = True
    else:
        raise "Missing options"

    if not skip_train:
        train_src = open(args.train_src).readlines()
        train_tgt = open(args.train_tgt).readlines()
        print(f'# train_src examples: {len(train_src)}')
        print(f'# train_tgt examples: {len(train_tgt)}')
        assert len(train_src) == len(train_tgt), 'Should be same len(train_src) and len(train_tgt).'
        if os.path.exists(args.out_train):
            os.remove(args.out_train)

    if not skip_valid:
        valid_src = open(args.valid_src).readlines()
        valid_tgt = open(args.valid_tgt).readlines()
        print(f'# valid_src examples: {len(valid_src)}')
        print(f'# valid_tgt examples: {len(valid_tgt)}\n')
        assert len(valid_src) == len(valid_tgt), 'Should be same len(valid_src) and len(valid_tgt).'
        if os.path.exists(args.out_valid):
            os.remove(args.out_valid)

    # create train.tsv
    if not skip_train:
        count = 0
        for src, tgt in zip(tqdm(train_src), train_tgt):
            src = src.replace('\n', '')
            tgt = tgt.replace('\n', '')
            src_words = src.split(' ')
            tgt_words = tgt.split(' ')
            if len(src_words) > args.maxlen or len(tgt_words) > args.maxlen:
                print('Remove sample:')
                print('SRC: ' + src)
                print('TGT: ' + tgt + '\n')
                count += 1
                continue
            if not src or not tgt:
                continue
            if src[:2] == '" ':
                src = src[2:]
            if tgt[:2] == '" ':
                tgt = tgt[2:]
            with open(args.out_train, 'a') as f:
                f.write(src + '\t' + tgt + '\n')
        print(f'Create {args.out_train}  ... {count} samples removed\n')

    # create valid.tsv
    if not skip_valid:
        for src, tgt in zip(tqdm(valid_src), valid_tgt):
            src = src.replace('\n', '')
            tgt = tgt.replace('\n', '')
            if src[:2] == '" ':
                src = src[2:]
            if tgt[:2] == '" ':
                tgt = tgt[2:]
            with open(args.out_valid, 'a') as f:
                f.write(src + '\t' + tgt + '\n')
        print(f'Create {args.out_valid}\n')


if __name__ == '__main__':
    main()
