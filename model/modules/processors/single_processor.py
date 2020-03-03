from transformers.data.processors.utils import DataProcessor, InputFeatures
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
from .consts import InputExample
from .text_processor import filter_text

class SingleProcessor(DataProcessor):
  
  def __init__(self, agent, label_names, task):
    assert agent == "therapist" or agent == "patient"
    assert task == "forecast" or task == "categorize"

    self.agent = 'T' if agent == 'therapist' else 'P'
    self.label_names = label_names
    self.task = task

  def get_examples(self, data_dir, mode):
    return self._create_examples(pd.read_json(os.path.join(data_dir, "{}.json".format(mode)), lines=True))

  def get_labels(self):
    """See base class."""
    return self.label_names 

  def _create_examples(self, df):
    """Creates examples for the training and dev sets.
       Task type is either forecasting or categorizing
       TODO implement forecating"""

    agent_ids = np.array([entry[0]["speaker"] == self.agent for entry in df["options-for-correct-answers"]], dtype="bool")
    df = df.iloc[agent_ids]

    examples = []
    for (_index, row) in df.iterrows():
      guid = row["example-id"]
      context, utterance = None, None
      if self.task == "forecast":
        context = filter_text(row["messages-so-far"][-1]["utterance"]) # example not used if no last context message
      else:
        utterance = filter_text(row["options-for-correct-answers"][0]["utterance"])
      label = row["options-for-correct-answers"][0]["agg_label"]
      if utterance or context:
        examples.append(InputExample(guid=guid, utterance=utterance, context=context, label=label))
    return examples
  

  def convert_examples_to_features(
  self,
  examples,
  tokenizer,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
      len_examples = len(examples)
      if ex_index % 10000 == 0:
        logger.info("Writing example %d/%d" % (ex_index, len_examples))
      
      if self.task == "categorize":
        inputs =  tokenizer.encode_plus(
          example.utterance,
          add_special_tokens=True, 
          max_length=max_length,
          pad_to_max_length=True,)
      else:
        inputs = tokenizer.encode_plus(
          example.context, 
          add_special_tokens=True, 
          max_length=max_length,
          pad_to_max_length=True,)
      input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]

      if output_mode == "classification":
        label = label_map[example.label]
      elif output_mode == "regression":
        label = float(example.label)
      else:
        raise KeyError(output_mode)

      if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        logger.info("label: %s (id = %d)" % (example.label, label))

      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
        )
      )

    return features
