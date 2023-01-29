import argparse
import json
import numpy as np
import string
import re

import torch
from torch import nn, Tensor
from transformers import EncoderDecoderModel, BertTokenizerFast

import tornado
from tornado.web import Application, RequestHandler, StaticFileHandler, HTTPServer


def str2title(str):

    str = string.capwords(str)
    str = str.replace(' - - - ', ' — ')
    str = str.replace(' - - ', ' – ')
    str = str.replace('( ', '(')
    str = str.replace(' )', ')')
    str = re.sub(r'(\w)\s+-\s+(\w)', r'\1-\2', str)
    str = re.sub(r'(\w|")\s+:', r'\1:', str)
    str = re.sub(r'"\s+([^"]+)\s+"', r'"\1"', str)
    return str


class GenerateTitleHandler(RequestHandler):

    def initialize(self, model, tokenizer, encoder_max_length, decoder_max_length, device, default_num_titles=5):

        self.model = model
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.device = device
        self.default_num_titles = default_num_titles


    def post(self):

        abstract = self.get_body_argument('abstract')
        temperature = max(1.0, float(self.get_body_argument('temperature', 1.5)))
        num_titles = min(20, max(1, int(self.get_body_argument('num_titles', self.default_num_titles))))

        input_token_ids = self.tokenizer(abstract, truncation=True, max_length=self.encoder_max_length, return_tensors='pt').input_ids.to(self.device)
        pred = self.model.generate(
            input_token_ids,
            decoder_start_token_id=self.tokenizer.cls_token_id, eos_token_id=self.tokenizer.sep_token_id, pad_token_id=self.tokenizer.pad_token_id,
            do_sample=(temperature > 1),
            num_beams=10,
            max_length=self.decoder_max_length,
            no_repeat_ngram_size=2,
            temperature=temperature,
            top_k=50,
            num_return_sequences=num_titles
        )
        titles = [str2title(title) for title in tokenizer.batch_decode(pred, True)]

        self.write(json.dumps(titles))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Web interface for BERT2BERT paper title generation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', type=str, help='Path to the model checkpoint to be loaded.')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased', help='The tokenizer to be used with the model.')
    parser.add_argument('--encoder-max-length', type=int, default=512, help='Maximum input length for the encoder.')
    parser.add_argument('--decoder-max-length', type=int, default=32, help='Maximum input length for the decoder.')
    parser.add_argument('--default-num', type=int, default=5, help='Default number of generated titles per request.')
    parser.add_argument('--port', type=int, default=8080, help='Webserver port.')
    parser.add_argument('--num-proc', type=int, default=1, help='Number of concurrent server processes.')
    parser.add_argument('--device', type=str, default=None, help='Computing device ("cuda" or "cpu").')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debugging mode.')
    args = parser.parse_args()

    print('Loading model...')
    device = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
    model = EncoderDecoderModel.from_pretrained(args.model).to(device)
    print('Ready.')
    
    app = Application([
        (r'/title', GenerateTitleHandler, {
            'model' : model,
            'tokenizer' : tokenizer,
            'device' : device,
            'encoder_max_length' : args.encoder_max_length,
            'decoder_max_length' : args.decoder_max_length,
            'default_num_titles' : args.default_num
        }),
        (r'/(.*)', StaticFileHandler, { 'path' : 'web', 'default_filename' : 'index.html' })
    ], debug=args.debug)
    if args.num_proc != 1:
        server = HTTPServer(app)
        server.listen(args.port)
        server.start(args.num_proc)
    else:
        app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
