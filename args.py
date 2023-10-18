# -*- coding:utf-8 -*-
import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='training epochs')
    parser.add_argument('--input_size', type=int, default=13, help='input dimension')
    parser.add_argument('--seq_len', type=int, default=24, help='seq len')
    parser.add_argument('--output_size', type=int, default=12, help='output dimension')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--out_channels', type=int, default=64, help='out channels')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    parser.add_argument('--lr', type=float, default=0.008, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--add_att', type=bool, default=False, help='add attention?')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='')
    parser.add_argument('--step_size', type=int, default=150, help='step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
    parser.add_argument('--num_nodes', type=int, default=0, help='')
    parser.add_argument('--file_path', type=str, default="data/PEMSD4/", help='')
    parser.add_argument('--motif_num', type=int, default=8, help='')

    # smegcn
    parser.add_argument('--N_exp', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
    parser.add_argument('--sme_num_layers', type=int, default=2)  # 64
    parser.add_argument('--early', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3000,
                        help="patience step for early stopping")  # 5e-4
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout for GCN")
    parser.add_argument('--embedding_dropout', type=float, default=0.6,
                        help='dropout for embeddings')

    parser.add_argument('--alpha', type=float, default=0.1,
                        help="residual weight for input embedding")
    parser.add_argument('--weight_decay1', type=float, default=0.01, help='weight decay in some models')
    parser.add_argument('--weight_decay2', type=float, default=5e-4, help='weight decay in some models')
    parser.add_argument('--type_norm', type=str, default="batch")
    parser.add_argument('--num_feats', type=int, default="13")
    parser.add_argument('--model_type', type=str, default="stgcn")

    args = parser.parse_args()

    return args
