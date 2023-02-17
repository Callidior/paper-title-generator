import os
import string
import re
import json
import requests
from typing import List, Optional

import torch
from transformers import EncoderDecoderModel, BertTokenizerFast

import gradio as gr


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


class Predictor:

    def __init__(
        self,
        model: EncoderDecoderModel,
        tokenizer: BertTokenizerFast,
        device: torch.device,
        num_titles: int,
        encoder_max_length: int = 512,
        decoder_max_length: int = 32,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_titles = num_titles
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length


    def __call__(self, abstract: str, temperature: float) -> List[str]:

        temperature = max(1.0, float(temperature))
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
            num_return_sequences=self.num_titles
        )
        titles = [str2title(title) for title in tokenizer.batch_decode(pred, True)]
        return titles


class HostedInference:

    def __init__(self, model: str, num_titles: int, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.model = model
        self.num_titles = num_titles
        self.api_key = api_key

    def __call__(self, abstract: str, temperature: float) -> List[str]:
        temperature = max(1.0, float(temperature))
        data = json.dumps({
            'inputs' : abstract,
            'parameters' : {
                'do_sample': (temperature > 1),
                'num_beams': 10,
                'temperature': temperature,
                'top_k': 50,
                'no_repeat_ngram_size': 2,
                'num_return_sequences': self.num_titles,
            },
            'options' : { 'use_cache' : False, 'wait_for_model' : True }
        })
        api_url = "https://api-inference.huggingface.co/models/" + self.model
        headers = { "Authorization": f"Bearer {self.api_key}" } if self.api_key is not None else {}
        response = requests.request("POST", api_url, headers=headers, data=data)
        response = json.loads(response.content.decode("utf-8"))

        if isinstance(response, dict) and ('error' in response):
            raise RuntimeError(response['error'])

        titles = [str2title(title['summary_text']) for title in response]
        return titles


def create_gradio_ui(predictor):

    inputs = [
        gr.Textbox(label="Paper Abstract", lines=10),
        gr.Slider(label="Creativity", minimum=1.0, maximum=2.5, step=0.1, value=1.5),
    ]
    outputs = ["text"] * predictor.num_titles

    description = (
        "<center>"
        "Bert2Bert model trained on computer science papers from arXiv to generate "
        "paper tiles from abstracts."
        "</center>"
    )

    ui = gr.Interface(
        fn=predictor,
        inputs=inputs,
        outputs=outputs,
        title="Paper Title Generator",
        description=description,
    )
    return ui


if __name__ == '__main__':
    model_path = "Callidior/bert2bert-base-arxiv-titlegen"

    if torch.cuda.is_available():
        print('Loading model...')
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = EncoderDecoderModel.from_pretrained(model_path).cuda()
        predictor = Predictor(model, tokenizer, device="cuda", num_titles=5)
        print(f'Ready - running on GPU.')
    else:
        print(f'No GPU available - using hosted inference API.')
        predictor = HostedInference(model_path, num_titles=5, api_key=os.environ.get("HF_TOKEN"))
    
    interface = create_gradio_ui(predictor)
    interface.launch()
