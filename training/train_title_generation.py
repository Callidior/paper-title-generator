import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn, Tensor
from transformers import (
    EncoderDecoderModel, AutoTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
)
from datasets import load_metric

from data import ArxivAbstractTitleDataset

from typing import Optional, Callable, List


rouge = load_metric('rouge')


def compute_metrics(pred):
    
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


class DataCollatorForBart(DataCollatorForSeq2Seq):
    """ Prepends eos_token_id to decoder_input_ids to simulate label shifting. """
    
    def __call__(self, features):
        
        features = super().__call__(features)
        features['decoder_input_ids'] = torch.cat([
            torch.LongTensor([[self.tokenizer.eos_token_id]]).expand(features['decoder_input_ids'].shape[0], 1),
            features['decoder_input_ids'][:, :-1]
        ], dim=1)
        return features


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trains a BERT2BERT model to generate titles from abstracts.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('arxiv_json', type=str, help='Path to the arXiv dataset JSON file.')
    parser.add_argument('save_dir', type=str, help='Directory where the trained model should be saved.')
    parser.add_argument('--base-model', type=str, default='bert-base-uncased', help='The base model used as initialization.')
    parser.add_argument('--encoder-max-length', type=int, default=512, help='Maximum input length for the encoder.')
    parser.add_argument('--decoder-max-length', type=int, default=32, help='Maximum input length for the decoder.')
    parser.add_argument('--test-split-size', type=float, default=0.1, help='Fraction of the data to be used for validation.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--warmup-steps', type=int, default=250, help='Learning rate warmup steps.')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory for logfiles and checkpoints.')
    args = parser.parse_args()

    model_name = args.base_model.split('/')[-1]
    output_dir = args.output_dir if args.output_dir is not None else f'./title-generation-arXiv-{model_name}'

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    data = ArxivAbstractTitleDataset(
        args.arxiv_json,
        tokenizer=tokenizer,
        encoder_max_length=args.encoder_max_length,
        decoder_max_length=args.decoder_max_length
    )

    if args.test_split_size > 0:
        train_ind, test_ind = train_test_split(np.arange(len(data)), test_size=args.test_split_size, random_state=0)
        data_train = torch.utils.data.Subset(data, train_ind)
        data_test = torch.utils.data.Subset(data, test_ind)
    else:
        data_train = data
        data_test = None
    
    if model_name.startswith('bart-'):
        CollatorClass = DataCollatorForBart
        model = BartForConditionalGeneration.from_pretrained(args.base_model).cuda()
    else:
        CollatorClass = DataCollatorForSeq2Seq
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.base_model, args.base_model).cuda()
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.encoder.vocab_size
    model.config.max_length = args.decoder_max_length
    model.config.min_length = 3
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.num_beams = 4

    train_args = Seq2SeqTrainingArguments(
        output_dir=f'{output_dir}/out',
        logging_dir=f'{output_dir}/logs',
        
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        fp16=True,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        dataloader_num_workers=8,
        
        do_eval=(data_test is not None),
        predict_with_generate=True,
        evaluation_strategy='no' if data_test is None else 'epoch',
        save_steps=0,
        save_total_limit=0,
        load_best_model_at_end=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=CollatorClass(tokenizer=tokenizer),
        args=train_args,
        compute_metrics=compute_metrics,
        train_dataset=data_train,
        eval_dataset=data_test,
    )
    trainer.train()

    model.save_pretrained(args.save_dir)
