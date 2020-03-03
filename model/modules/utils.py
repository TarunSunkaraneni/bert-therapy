from sklearn.metrics import f1_score
from processors.single_processor import SingleProcessor
from processors.context_processor import ContextProcessor
from processors.speaker_context_processor import SpeakerContextProcessor
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# https://github.com/utahnlp/therapist-observer/blob/3ed5332df1b2d761868e2519da1099834c963c90/tensorflow/psyc_utils.py
MISC11_P_labels = ["change_talk","sustain_talk","follow_neutral"]
MISC11_T_labels = ["facilitate","reflection_simple","reflection_complex","giving_info","question_closed","question_open","MI_adherent","MI_non-adherent"]

MISC11_BRIEF_T_labels = ["FA","RES","REC","GI","QUC","QUO","MIA","MIN"]
MISC11_BRIEF_P_labels = ["POS","NEG","FN"]

processors = {
  "forecast-therapist-single": SingleProcessor("therapist", MISC11_T_labels, "forecast"),
  "forecast-patient-single": SingleProcessor("patient", MISC11_P_labels, "forecast"),
  "forecast-therapist-concat": ContextProcessor("therapist", MISC11_T_labels, "forecast", context_len=9, concat_context=True),
  "forecast-patient-concat": ContextProcessor("patient", MISC11_P_labels, "forecast", context_len=9, concat_context=True),
  ####
  "categorize-therapist-single": SingleProcessor("therapist", MISC11_T_labels, "categorize"),
  "categorize-patient-single": SingleProcessor("patient", MISC11_P_labels, "categorize"),
  "categorize-therapist-concat": ContextProcessor("therapist", MISC11_T_labels, "categorize", context_len=9, concat_context=True),
  "categorize-patient-concat": ContextProcessor("patient", MISC11_P_labels, "categorize", context_len=9, concat_context=True),
  "categorize-therapist-speaker": SpeakerContextProcessor("therapist", MISC11_T_labels, "categorize", context_len=9, concat_context=False),
  "categorize-patient-speaker": SpeakerContextProcessor("patient", MISC11_P_labels, "categorize", context_len=9, concat_context=False)
  }

output_modes = defaultdict(lambda: "classification")

def compute_metrics(preds, labels):
  results = {"macro": f1_score(labels, preds, average="macro")}
  category_f1s = f1_score(labels, preds, average=None)
  label_names = MISC11_BRIEF_T_labels if len(category_f1s) > 3 else  MISC11_BRIEF_P_labels
  for cat, score in zip(label_names, category_f1s):
    results[cat] = score
  return results