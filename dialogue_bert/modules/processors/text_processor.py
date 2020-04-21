import re
from functools import lru_cache
import torch
from torchtext.data import TabularDataset
from transformers import RobertaTokenizer, BertTokenizer

speaker_dict = {'P': 'patient', 'T': 'therapist'}

@lru_cache(maxsize=100)
def lru_encode(tokenizer, sentence):
  return tokenizer.encode(sentence)

@lru_cache(maxsize=100, typed=True)
def filter_text(text):
    text = re.sub(r"(\[\d*:*\d*\])", "", text) # removed timestamps
    paren_matches = re.findall(r"(\(.+?\))", text) + re.findall(r"(\[.+?\])", text)
    for match in paren_matches:
      t = match[1:-1].strip() # don't want the open and close braces  
      if len(t.split()) == 1 and t != 'du' and t != 'sp': # this is a code
        text = text.replace(match, "( {} )".format(t)) # add only the first verb lemmatized in the sequence
      else:
        text = text.replace(match, "")
    return re.sub(r'\[\]|\(\)', '', text).strip() # remove unnecessary tags and spaces

class PsychDataset(torch.utils.data.Dataset):
  def __init__(self, tabular_dataset):
    if isinstance(tabular_dataset, list):
      self.data = tabular_dataset
    elif isinstance(tabular_dataset, TabularDataset):
      self.data = [{**tabular_dataset[i].__dict__['context'], 
                    **tabular_dataset[i].__dict__['utterance']} for i in range(len(tabular_dataset))]
    else:
      raise NotImplementedError()

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    return self.data[index]
  
  def __add__(self, other):
    return self.data + other.data
  
  @staticmethod
  def load(file):
    return PsychDataset(torch.load(file))
  
  @staticmethod
  def save(dataset, directory):
    torch.save(dataset.data, directory) 


def utterance_processor(tokenizer):
  def _closure(x):
    x = x[0]
    x['utterance'] = filter_text(x['utterance'])
    return {'utterance': x['utterance'], 
            'utterance_encoded': tokenizer.encode(x['utterance']) if x['utterance'] else [],
            'utterance_speaker': x['speaker'],
            'utterance_label': x['agg_label'],
            'utterance_uid': x['uid']}
  return _closure

def context_processor(tokenizer):
  def _closure(x):
    context = []
    context_encoded = []
    speaker = []

    for turn in x:
      if turn['speaker'] == 'PAD':
        continue
      turn['utterance'] = filter_text(turn['utterance'])
      if turn['utterance']: # could be empty
        context.append(turn['utterance'])
        context_encoded.append(lru_encode(tokenizer, turn['utterance']))
        speaker.append(turn['speaker'])

    return {'context': context, 
            'context_encoded': context_encoded,
            'context_speaker': speaker}
  return _closure