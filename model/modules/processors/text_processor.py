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

safe_words = set(aliased.values())
safe_words.add("um")
safe_words.add("uh")
safe_words.add("like")
safe_words.add("laugh")

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
    elif action_word in safe_words:
      text_cleaned = text_cleaned.replace(action, "( {} )".format(action_word))
    else:
      text_cleaned = text_cleaned.replace(action, "")
  return text_cleaned.strip()

def encode_context(context, utterance_encode_len, max_len, etc_tokens, tokenizer):
  '''<s> hello what is up with you</s></s> I am fine</s>'''
  # note the spaces
  if isinstance(tokenizer, RobertaTokenizer):
    max_con_length, leftover = (max_len - utterance_encode_len) // len(context), 0
    for idx, con in enumerate(context):
      con_encoded = (
        ([tokenizer.bos_token_id] if idx == 0 else [tokenizer.sep_token_id]) +
        tokenizer.encode(con, add_special_tokens=False) +
        [tokenizer.eos_token_id])
      if len(con_encoded) < max_con_length:
        leftover += max_con_length - len(con_encoded)
      else:
        con_encoded = (
          [con_encoded[0]] +
          etc_tokens +
          con_encoded[1: min(max_con_length + leftover, len(con_encoded)-2)] + 
          [con_encoded[-1]])
        leftover = max(0, leftover - (max_con_length - len(con_encoded)))
      assert (
        len(con_encoded) > 2 and
        (con_encoded[0] == tokenizer.bos_token_id or con_encoded[0] == tokenizer.sep_token_id) and 
        con_encoded[-1] == tokenizer.eos_token_id
      )
      con_decoded = tokenizer.decode(con_encoded)
      context[idx] = con_decoded

  '''<s>[CLS] hello what is up with you [SEP]'''
  if isinstance(tokenizer, BertTokenizer):
    raise NotImplementedError()
  return ''.join(context)