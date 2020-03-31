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
    assert agent == "therapist" or agent == "patient" or agent == "both"
    assert task == "forecast" or task == "categorize"
    if agent == "both":
      assert isinstance(label_names, dict)
    else:
      assert isinstance(label_names, list)
    
    if agent == 'therapist':
      self.agent = 'T' 
    elif agent == 'patient':
      self.agent = 'P'
    else:
      self.agent = agent
    self.label_names = label_names
    self.task = task

  def get_examples(self, data_dir, mode):
    return self._create_examples(pd.read_json(os.path.join(data_dir, "{}.json".format(mode)), lines=True))

  def get_labels(self):
    """See base class."""
    if self.agent == "both":
      # second argument for masking logits in combined model
      return [label for labels in self.label_names.values() for label in labels]
    return self.label_names

  def _create_examples(self, df):
    """Creates examples for the training and dev sets.
       Task type is either forecasting or categorizing
    """
    if self.agent != "both":
      label_offset = defaultdict(int) # always 0
      agent_ids = np.array([entry[0]["speaker"] == self.agent for entry in df["options-for-correct-answers"]], dtype="bool")
      df = df.iloc[agent_ids]
    else:
      label_offset, offset = {}, 0
      for agent, labels in self.label_names.items():
        label_offset[agent] = offset
        offet += len(labels)

    examples = []
    for (_index, row) in df.iterrows():
      guid, history, current = row["example-id"], row["messages-so-far"], row["options-for-correct-answers"]
      context, utterance = "", ""
      if self.task == "forecast":
        if history[-1]["speaker"] == "PAD":
          continue
        context = filter_text(history[-1]["utterance"])
      else:
        utterance = filter_text(current[0]["utterance"])
      label = current[0]["agg_label"] + label_offset[current[0]["speaker"]]
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
      text = example.utterance if self.task == "categorize" else example.context
      inputs =  tokenizer.encode_plus(
        example.utterance,
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
        logger.info("%s" % tokenizer.decode(input_ids))
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
