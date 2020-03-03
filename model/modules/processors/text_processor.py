import re
from functools import lru_cache
from transformers import RobertaTokenizer, BertTokenizer

aliased = {'Pause': 'pause', 'laughs': 'laugh', 
'chuckles': 'chuckle', 'chuckling': 'chuckle', 'PAUSE': 'pause', 'laughter': 'laugh', 'LAUGHTER': 'laugh',
'Laughter': 'laugh', 'Silence': 'silence', 'laughing': 'laugh', 'Crosstalk': 'crosstalk', 'unintelligible': 'unclear', 
'Sigh': 'sigh', 'cross talk': 'crosstalk', 'Sniff': 'sniff', 'Laughing': 'laugh', 'SIGH': 'sigh', 'Unclear': 'unclear',
'Chuckling': 'chuckle', 'coughs': 'cough', 'Clears throat': 'clear throat', 'sniggers': 'snicker', 'crosstalking': 'crosstalk',
 'sniffles': 'sniff', 'PH': 'ph', 'inaudible at': 'inaudible', 'Long pause': 'long pause', 'sniffling': 'sniff',
'Sniffles': 'sniff', 'giggles': 'giggle', 'coughing': 'cough', 'sighing': 'sigh', 'sniffs': 'sniff', 'whispers': 'whisper',
 'Sighs': 'sigh', 'Cross talk': 'crosstalk', '3 second pause': 'pause', 'CROSSTALK': 'crosstalk',
'2 second pause': 'pause', 'whispering': 'whisper', 'Interposing': 'interpose', 'Crying': 'cry', 'Sniffling': 'sniff',
'cries': 'cry', 'Coughs': 'cough', 'Unintelligible': 'unclear'} 

safe_verbs = set(aliased.values())
safe_verbs.add("um")
safe_verbs.add("like")
safe_verbs.add("laugh")

@lru_cache(maxsize=50, typed=True)
def filter_text(text):
  if not text:
    return text
  text_cleaned = re.sub(r"(\[\d*:*\d*\])", "", text) # removed timestamps
  actions = re.findall(r"(\(.+?\))", text_cleaned) + re.findall(r"(\[.+?\])", text_cleaned)
  for action in actions:
    action_word = "".join(x for x in action if x.isalpha())
    if action_word in aliased:
      text_cleaned = text_cleaned.replace(action, "( {} )".format(aliased.get(action_word, action_word)))
    elif action_word in safe_verbs:
      text_cleaned = text_cleaned.replace(action, "( {} )".format(action_word))
    else:
      text_cleaned = text_cleaned.replace(action, "")
  return text_cleaned.strip()

def encode_context(context, utterance_encode_len, max_len, etc_token, tokenizer):
  '''<s> hello what is up with you</s></s> I am fine</s>'''
  # note the spaces
  if isinstance(tokenizer, RobertaTokenizer):
    max_con_length, leftover = (max_len - utterance_encode_len) // len(context), 0
    for idx, con in enumerate(context):
      con_encoded = tokenizer.encode(con)
      if idx != 0:
        con_encoded[0] = tokenizer.sep_token_id
      if len(con_encoded) < max_con_length:
        leftover += max_con_length - len(con_encoded)
      elif len(con_encoded) < max_con_length + leftover:
        leftover -= (len(con_encoded) - max_con_length)
      else:
        con_encoded = [con_encoded[0]]+ etc_token + con_encoded[(len(con_encoded)-max_con_length-leftover)+len(etc_token)+1:]
        leftover = 0
      con_decoded = tokenizer.decode(con_encoded)
      # HACK: this is to add a space for roberta's first sentence
      con_decoded = con_decoded[:len(tokenizer.bos_token)] + ' ' + con_decoded[len(tokenizer.bos_token):]
      context[idx] = con_decoded

  '''<s>[CLS] hello what is up with you [SEP]'''
  if isinstance(tokenizer, BertTokenizer):
    raise NotImplementedError()
  return ''.join(context)