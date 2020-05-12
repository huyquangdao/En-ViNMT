from base.dataset import BaseDataset
import numpy as np
from utils.data_utils import pad_to_max_length

class NMTDataset(BaseDataset):

    def __init__(self, src_corpus_path, des_corpus_path, src_tokenizer, des_tokenizer, max_seq_length):

        super(NMTDataset,self).__init__()

        with open(src_corpus_path,'r') as f:
            self.src_lines = f.readlines()
        with open(des_corpus_path,'r') as f:
            self.des_lines = f.readlines()
        
        self.src_tokenizer = src_tokenizer
        self.des_tokenizer = des_tokenizer

        self.max_seq_length = max_seq_length

        assert len(self.src_lines) == len(self.des_lines)
    
    def __len__(self):
        return len(self.src_lines)
    
    def __getitem__(self, idx):

        src, des = self.src_lines[idx], self.des_lines[idx]

        src, des = self.src_tokenizer.encode(src), self.des_tokenizer.encode(des)

        src_ids, des_ids = src.ids, des.ids

        padded_src_ids = pad_to_max_length(src_ids, self.src_tokenizer, self.max_seq_length)
        padded_des_ids = pad_to_max_length(des_ids, self.des_tokenizer, self.max_seq_length)

        assert len(padded_src_ids) == self.max_seq_length
        assert len(padded_des_ids) == self.max_seq_length

        # print(len(padded_src_ids))
        # print(len(padded_des_ids))

        return np.array(padded_src_ids), np.array(padded_des_ids)