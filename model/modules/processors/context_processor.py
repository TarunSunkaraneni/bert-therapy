from transformers.data.processors.utils import DataProcessor, InputFeatures
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import re
import logging

logger = logging.getLogger(__name__)
from .consts import InputExample
from .text_processor import filter_text, encode_context

class ContextProcessor(DataProcessor):
  
  def __init__(self, agent, label_names, task, context_len=1, concat_context=True):
    assert agent == "therapist" or agent == "patient"
    assert task == "forecast" or task == "categorize"
    assert context_len > 0

    self.agent = 'T' if agent == 'therapist' else 'P'
    self.label_names = label_names
    self.task = task # "categorize" or "forecast"
    self.context_len = context_len # length of context.
    self.concat_context = concat_context # True or False

  def get_examples(self, data_dir, mode):
    return self._create_examples(pd.read_json(os.path.join(data_dir, "{}.json".format(mode)), lines=True))

  def get_labels(self):
    """See base class."""
    return self.label_names 

  def _create_examples(self, df):
    """Creates examples for the training and dev sets.
       Task type is either forecasting or categorizing"""
    agent_ids = np.array([entry[0]["speaker"] == self.agent for entry in df["options-for-correct-answers"]], dtype="bool")
    df = df.iloc[agent_ids]

    examples = []
    for (_index, row) in df.iterrows():
      guid = row["example-id"]
      utterance = ""
      """HACK: reversing context so the truncation strategy works in encode_plus"""
      reversed_context = [filter_text(row["messages-so-far"][-i]["utterance"]) for i in range(self.context_len, 1, -1)]
      if self.task == "categorize":
        utterance = filter_text(row["options-for-correct-answers"][0]["utterance"])
      label = row["options-for-correct-answers"][0]["agg_label"]
      examples.append(InputExample(guid=guid, utterance=utterance, context=reversed_context, label=label))
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
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
      examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
      tokenizer: Instance of a tokenizer that will tokenize the examples
      max_length: Maximum example length
      label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
      output_mode: String indicating the output mode. Either ``regression`` or ``classification``
      pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
      pad_token: Padding token
      pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
      mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
        and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
        actual values)
    Returns:
      given input of a list of ``InputExamples``, will return
      a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
      len_examples = len(examples)
      if ex_index % 10000 == 0:
        logger.info("Writing example %d/%d" % (ex_index, len_examples))
      if not example.utterance.strip():
        continue
      utterance = tokenizer.decode(tokenizer.encode(example.utterance))
      etc_token = tokenizer.encode('...', add_special_tokens=False)
      if self.concat_context:
        context = tokenizer.encode(' '.join(example.context))
        if len(context) > max_length:
          context = context[0]+ etc_token + context[(len(context)-max_length)+len(etc_token)+1:]
        context = tokenizer.decode(context)
        # HACK: this is to add a space for roberta's first sentence
        context = context[:len(tokenizer.bos_token)] + ' ' + context[len(tokenizer.bos_token):]
      else:
        context = encode_context(example.context, len(utterance), max_length, etc_token, tokenizer)
        # we do not need to truncate after this.
        if self.task == "categorize":
          inputs = tokenizer.encode_plus(
            context,
            utterance,
            add_special_tokens=False, 
            max_length=max_length,
            pad_to_max_length=True,
            truncation_strategy='do_no_truncate')
        else:
          inputs = tokenizer.encode_plus(
          context, 
          add_special_tokens=False, 
          max_length=max_length,
          pad_to_max_length=True,
          truncation_strategy='do_no_truncate')
      
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
