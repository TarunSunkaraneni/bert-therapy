from sklearn.metrics import f1_score
from processors.speaker_context_processor import SpeakerContextProcessor
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# https://github.com/utahnlp/therapist-observer/blob/3ed5332df1b2d761868e2519da1099834c963c90/tensorflow/psyc_utils.py
MISC11_P_labels = ["change_talk","sustain_talk","follow_neutral"]
MISC11_T_labels = ["facilitate","reflection_simple","reflection_complex","giving_info","question_closed","question_open","MI_adherent","MI_non-adherent"]

MISC11_BRIEF_T_labels = ["FA","RES","REC","GI","QUC","QUO","MIA","MIN"]
MISC11_BRIEF_P_labels = ["POS","NEG","FN"]

# https://forums.fast.ai/t/focalloss-with-multi-class/35588/3  
def one_hot_embedding(labels, num_classes):
    return (torch.eye(num_classes)[labels]).to(labels.device)

class FocalLoss(nn.Module):
  def __init__(self, alpha, gamma=0, eps=1e-7):
    super(FocalLoss, self).__init__()
    self.alpha = torch.FloatTensor(alpha)
    self.gamma = gamma
    self.eps = eps

  def forward(self, logits, labels):
    y = one_hot_embedding(labels, logits.size(-1))
    probs= F.softmax(logits, dim=-1)
    probs = probs.clamp(self.eps, 1. - self.eps)
    alphas = self.alpha[labels].unsqueeze(1).to(labels.device)
    loss = -1 * alphas * y * torch.log(probs) # alpha-weighted cross entropy
    loss = loss * (1 - probs) ** self.gamma # focal loss
    return loss.sum(dim=1).mean()

processors = {
  "categorize-both-speaker-dialogue-roberta": SpeakerContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    "categorize", 
    context_len=9),
  "categorize-both-speaker-conversational-bert": SpeakerContextProcessor(
    {"P": MISC11_P_labels, 
     "T": MISC11_T_labels}, 
    "categorize", 
    context_len=9),
  "categorize-both-speaker-conversational-bert-fl": SpeakerContextProcessor(
    {"P": MISC11_P_labels, 
      "T": MISC11_T_labels}, 
    "categorize", 
    context_len=9,
    loss_function= FocalLoss([1.0, 1.0, 0.25, 0.5, 1.0, 1.0, 1.0, 0.75, 0.75, 1.0, 1.0], gamma=1.0)
  )
}

def compute_metrics(preds, labels, label_names):
  results = {"macro": f1_score(labels, preds, average="macro")}
  category_f1s = f1_score(labels, preds, average=None)
  for cat, score in zip(label_names, category_f1s):
    results[cat] = score
  return results


""" Binary multi task logits masker"""
sep = {name: processor.label_dict['_SEP'] for name, processor in processors.items()}
def logits_masked(logits, labels, task_name):
  s = sep[task_name]
  m_logits = logits.clone()
  m_logits[(labels < s).view((-1, )), s:]= -10000
  m_logits[(labels >= s).view((-1, )), :s] = -10000
  return m_logits