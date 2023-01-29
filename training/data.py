import numpy as np
import newlinejson
import os.path

import torch
from torch import nn, Tensor
from transformers import PreTrainedTokenizerBase

from typing import Optional, Union, List, Tuple, Callable


class Seq2SeqDataset:

    def __init__(self,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 decoder_tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 transform: Optional[Callable] = None,
                 encoder_max_length: Optional[int] = None,
                 decoder_max_length: Optional[int] = None) -> None:
        
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else self.tokenizer
        self.transform = transform
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.samples = []
    
    
    def __getitem__(self, idx: int):
        
        seq_in, seq_out = self.get_sample(idx)
        if self.tokenizer is None:

            if self.transform:
                seq_in, seq_out = self.transform(seq_in, seq_out)
            return seq_in, seq_out

        else:

            data = self.tokenizer(seq_in,
                                  truncation=(self.encoder_max_length is not None),
                                  padding='max_length' if self.encoder_max_length is not None else False,
                                  max_length=self.encoder_max_length)
            outputs = self.decoder_tokenizer(seq_out,
                                             truncation=(self.decoder_max_length is not None),
                                             padding='max_length' if self.decoder_max_length is not None else False,
                                             max_length=self.decoder_max_length)
            data['decoder_input_ids'] = outputs['input_ids']
            data['decoder_attention_mask'] = outputs['attention_mask']
            data['labels'] = outputs['input_ids'].copy()
            if self.transform:
                data = self.transform(data)
            return data
    
    
    def __len__(self) -> int:
        
        return len(self.samples)


    def get_sample(self, idx: int) -> Tuple[str, str]:

        return self.samples[idx]


class ArxivAbstractTitleDataset(Seq2SeqDataset):
    
    def __init__(self,
                 json_file: str,
                 categories: Union[str, List[str]] = ['cs.', 'stat.ML'],
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 decoder_tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 transform: Optional[Callable] = None,
                 encoder_max_length: Optional[int] = None,
                 decoder_max_length: Optional[int] = None) -> None:
        
        super().__init__(tokenizer=tokenizer,
                         decoder_tokenizer=decoder_tokenizer,
                         transform=transform,
                         encoder_max_length=encoder_max_length,
                         decoder_max_length=decoder_max_length)

        self.json_file = json_file
        self.categories = [categories] if isinstance(categories, str) else categories
        
        with newlinejson.open(self.json_file) as f:
            for paper in f:
                if (self.categories is None) or any(any(papercat.startswith(cat) for cat in self.categories) for papercat in paper['categories'].split()):
                    self.samples.append((paper['title'], paper['abstract']))
