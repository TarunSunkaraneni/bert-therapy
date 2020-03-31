from transformers.data.processors.utils import DataProcessor, InputFeatures
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import re
import logging

logger = logging.getLogger(__name__)
from .consts import InputExample
from .text_processor import filter_text
from .context_processor import ContextProcessor

class SpeakerContextProcessor(ContextProcessor):
  def _create_examples(self, df):
    """Creates examples for the training and dev sets.
        Task type is either forecasting or categorizing"""
    if self.agent != "both":
      agent_ids = np.array([entry[0]["speaker"] == self.agent for entry in df["options-for-correct-answers"]], dtype="bool")
      df = df.iloc[agent_ids]

    examples = []
    speaker_code_dict = {"T": "therapist", "P": "patient"}
    for (_index , row) in df.iterrows():
      guid, history, current = row["example-id"], row["messages-so-far"], row["options-for-correct-answers"]
      utterance = ""
      if all([history[i]["speaker"] == "PAD" for i in self.context_range]):
        context = ["no context"]
      else:
        context = ["{}: {}".format(
          speaker_code_dict[history[i]["speaker"]], 
          filter_text(history[i]["utterance"])) 
        for i in self.context_range if history[i]["speaker"] != "PAD"]
      if self.task == "categorize":
        if self.agent == "both":
          utterance = "{} utter: {}".format(
            speaker_code_dict[current[0]["speaker"]], 
            filter_text(current[0]["utterance"]))
        else:
          utterance = "utter: {}".format(
            filter_text(current[0]["utterance"]))
      label = current[0]["agg_label"]
      examples.append(InputExample(guid=guid, utterance=utterance, context=context, label=label))
    return examples