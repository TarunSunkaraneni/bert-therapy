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
    agent_ids = np.array([entry[0]["speaker"] == self.agent for entry in df["options-for-correct-answers"]], dtype="bool")
    df = df.iloc[agent_ids]

    examples = []
    speaker_code_dict = {"T": "therapist", "P": "patient", "PAD": ""}
    for (_index, row) in df.iterrows():
      guid = row["example-id"]
      utterance = ""
      context = ["{} : {}".format(speaker_code_dict[row["messages-so-far"][-i]["speaker"]], filter_text(row["messages-so-far"][-i]["utterance"])) 
        for i in range(self.context_len, 1, -1)]
      if self.task == "categorize":
        utterance = filter_text(row["options-for-correct-answers"][0]["utterance"])
      label = row["options-for-correct-answers"][0]["agg_label"]
      examples.append(InputExample(guid=guid, utterance=utterance, context=context, label=label))
    return examples