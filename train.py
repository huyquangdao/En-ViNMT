import torch.nn as nn
import torch.optim as optim
import argparse
import torch

from trainers.DogCatTrainer import DogCatTrainer
from models.DogCatModel import Resnet34

from metrics.classification_metric import ClassificationMetric
from utils.utils import set_seed

from dataset.nmt_dataset import NMTDataset
from utils.log import Writer

import torch.optim as optim
from models.transfomers import Encoder, Decoder, Seq2Seq
from utils.data_utils import create_tokenizer
from utils.utils import initialize_weights

from trainers.nmt_trainer import NMTTrainer


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_src_file', help='Your training directory', default='data/train/train.en')
    parser.add_argument('--train_des_file', help='Your training directory', default='data/train/train.vi')
    parser.add_argument('--test_src_file', help='Your testing directory', default='data/test/tst2013.en')
    parser.add_argument('--test_des_file', help='Your testing directory', default='data/test/tst2013.vi')


    parser.add_argument('--batch_size',help='Your training batch size',default=16, type = int)
    parser.add_argument('--num_workers', help='number of process', default=2, type = int)
    parser.add_argument('--seed',help='random seed',default=1234, type= int)
    parser.add_argument('--epoch', help='training epochs', default=5, type = int)
    parser.add_argument('--lr',help='learning rate',default=0.001)
    parser.add_argument('--max_lr', help = 'maximum learning rate', default=0.01, type= float)
    parser.add_argument('--val_batch_size', help='Your validation batch size', default=8)
    parser.add_argument('--grad_clip',help='gradient clipping theshold',default=5, type = int)
    parser.add_argument('--grad_accum_step', help='gradient accumalation step', default=1)

    parser.add_argument('--n_classes',help='Number of classes', default=2, type=int)

    parser.add_argument('--pretrained',help='path to pretrained model', default=1, type=bool)
    parser.add_argument('--gpu',help='Number of classes', default=1, type= bool)

    parser.add_argument('--log_dir',help='Log directory path', default='logs', type= str)

    parser.add_argument('--lr_scheduler',help= 'learning rate scheduler', default = 'cyclic')
    parser.add_argument('--n_layers', help ='number of transfomer layer', default = 6, type = int)
    parser.add_argument('--n_heads', help= 'number of attention head', default = 8, type= int)

    parser.add_argument('--pf_dim', help ='position feedforward dimesion', default = 2048, type= int)
    parser.add_argument('--hidden_size', help= 'hidden_size', default = 512, type= int)
    parser.add_argument('--drop_out', help = 'drop out prop', default = 0.1, type= float)
    parser.add_argument('--max_seq_length', help='max sequence length', default = 100, type = int)
    
    parser.add_argument('--vocab_size', help='vocab size', default = 20000)

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()

    src_tokenizer = create_tokenizer(corpus_file_path = args.train_src_file, vocab_size = args.vocab_size)
    des_tokenizer = create_tokenizer(corpus_file_path = args.train_des_file, vocab_size = args.vocab_size)

    train_dataset = NMTDataset(src_corpus_path=args.train_src_file, des_corpus_path=args.train_des_file, src_tokenizer= src_tokenizer, des_tokenizer=des_tokenizer, max_seq_length= args.max_seq_length)

    if args.test_src_file != '':
        test_dataset = NMTDataset(src_corpus_path=args.test_src_file, des_corpus_path=args.test_des_file, src_tokenizer= src_tokenizer, des_tokenizer=des_tokenizer, max_seq_length= args.max_seq_length)


    criterion = nn.CrossEntropyLoss()
    metric = ClassificationMetric(n_classes=args.n_classes)

    # metric = None

    writer = Writer(log_dir=args.log_dir)

    if args.gpu:
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    encoder = Encoder(input_dim=args.vocab_size, hidden_size = args.hidden_size, n_layers= args.n_layers, n_heads = args.n_heads, pf_dim= args.pf_dim, drop_out= args.drop_out, device = DEVICE, max_length= args.max_seq_length)
    decoder = Decoder(output_dim=args.vocab_size, hidden_size= args.hidden_size, n_layers= args.n_layers , n_heads = args.n_heads, pf_dim = args.pf_dim, drop_out = args.drop_out, device = DEVICE, max_length=args.max_seq_length)

    src_pad_idx = src_tokenizer.token_to_id('<PAD>')
    des_pad_idx = des_tokenizer.token_to_id('<PAD>')

    model = Seq2Seq(encoder=encoder,decoder=decoder, src_pad_idx=src_pad_idx, trg_pad_idx=des_pad_idx, device = DEVICE)

    model.apply(initialize_weights)


    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer = optimizer,base_lr=args.lr,max_lr=args.max_lr,step_size_up=2000,cycle_momentum= False)

    trainer = NMTTrainer(model= model,
                        optimizer= optimizer,
                        criterion= criterion,
                        metric=metric,
                        log = writer,
                        lr_scheduler = lr_scheduler,
                        device = DEVICE,
                        )

    trainer.train(train_dataset=train_dataset,
                  epochs=args.epoch,
                  gradient_accumalation_step=args.grad_accum_step,
                  train_batch_size=args.batch_size,
                  num_workers=0,
                  gradient_clipping=args.grad_clip,
                  dev_dataset=test_dataset,
                  dev_batch_size = args.val_batch_size)









