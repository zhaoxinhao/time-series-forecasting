# -*- coding:utf-8 -*-

from args import args_parser
from get_data import nn_seq, setup_seed
from util import test, train

setup_seed(42)


def main():
    args = args_parser()
    #args.model_type = 'stgcn'  # 可以改为st-sme
    args.model_type = 'st-sme'
    args.file_path = 'data/PEMSD8/'  # 可以改为data/PEMSD3等
    args.motif_num = 7  # 3:4 4:6 7:3 8:8
    Dtr, Val, Dte, scaler, edge_index = nn_seq(args)
    print(len(Dtr), len(Val), len(Dte))
    # 训练完毕后下次可以只运行test
    train(args, Dtr, Val, edge_index)
    test(args, Dte, scaler, edge_index)


if __name__ == '__main__':
    main()
