import argparse
import json
import numpy as np
import string
import re
import requests
from distutils.util import strtobool

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

    def initialize(self, api_key, model, default_num_titles=5):

        self.api_key = api_key
        self.model = model
        self.default_num_titles = default_num_titles


    def post(self):

        abstract = self.get_body_argument('abstract')
        temperature = max(1.0, float(self.get_body_argument('temperature', 1.5)))
        num_titles = min(20, max(1, int(self.get_body_argument('num_titles', self.default_num_titles))))
        blocking = bool(strtobool(self.get_body_argument('blocking', True)))
        
        response = self.query_api(
            abstract,
            wait=blocking,
            do_sample=(temperature > 1),
            num_beams=10,
            temperature=temperature,
            top_k=50,
            no_repeat_ngram_size=2,
            num_return_sequences=num_titles
        )

        result = { 'titles' : [] }
        if isinstance(response, dict) and ('error' in response):
            # 'error' : 'Model Callidior/bert2bert-base-arxiv-titlegen is currently loading'
            # 'estimated_time': 19.793826919999997
            result['error'] = response['error']
        else:
            result['titles'] = [str2title(title['summary_text']) for title in response]

        self.write(json.dumps(result))


    def query_api(self, inputs, cache=False, wait=False, **kwargs):

        data = json.dumps({
            'inputs' : inputs,
            'parameters' : kwargs,
            'options' : { 'use_cache' : cache, 'wait_for_model' : wait }
        })
        api_url = "https://api-inference.huggingface.co/models/" + self.model
        headers = { "Authorization": f"Bearer {self.api_key}" }
        response = requests.request("POST", api_url, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Web interface for BERT2BERT paper title generation.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('api_key', type=str, help='Key for the Hugging Face Hosted Inference API.')
    parser.add_argument('model', type=str, help='Name of the hosted-inference model on the Hugging Face model repository.')
    parser.add_argument('--default-num', type=int, default=5, help='Default number of generated titles per request.')
    parser.add_argument('--port', type=int, default=8080, help='Webserver port.')
    parser.add_argument('--num-proc', type=int, default=1, help='Number of concurrent server processes.')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debugging mode.')
    args = parser.parse_args()
    
    app = Application([
        (r'/title', GenerateTitleHandler, {
            'api_key' : args.api_key,
            'model' : args.model,
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
