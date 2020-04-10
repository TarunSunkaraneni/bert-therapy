import argparse
import glob
import json
import logging
import os
import random
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm, trange

from transformers import (
  WEIGHTS_NAME,
  AdamW,
  AlbertConfig,
  AlbertForSequenceClassification,
  AlbertTokenizer,
  BertConfig,
  BertForSequenceClassification,
  BertTokenizer,
  DistilBertConfig,
  DistilBertForSequenceClassification,
  DistilBertTokenizer,
  FlaubertConfig,
  FlaubertForSequenceClassification,
  FlaubertTokenizer,
  RobertaConfig,
  RobertaForSequenceClassification,
  RobertaTokenizer,
  XLMConfig,
  XLMForSequenceClassification,
  XLMRobertaConfig,
  XLMRobertaForSequenceClassification,
  XLMRobertaTokenizer,
  XLMTokenizer,
  XLNetConfig,
  XLNetForSequenceClassification,
  XLNetTokenizer,
  get_linear_schedule_with_warmup,
)

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
  (
    tuple(conf.pretrained_config_archive_map.keys())
    for conf in (
      BertConfig,
      XLNetConfig,
      XLMConfig,
      RobertaConfig,
      DistilBertConfig,
      AlbertConfig,
      XLMRobertaConfig,
      FlaubertConfig,
    )
  ),
  (),
)

MODEL_CLASSES = {
 "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
  "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
  "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
  "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
  "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
  "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
  "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
  "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer)
}

SPECIAL_TOKENS = ["<s>", "<eos>", "<therapist>", "<client>", "<utterance>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>',
                         'additional_special_tokens': ["<therapist>", "<patient>", "<utterance>", "</therapist>", "</patient>", "</utterance>"]}
symbol_dict = dict()

from utils import compute_metrics, processors, output_modes, logits_masked

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

def add_special_tokens_(model, tokenizer):
  """ Add special tokens to the tokenizer and the model if they have not already been added. """
  orig_num_tokens = len(tokenizer.encoder)
  num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
  if num_added_tokens > 0:
      model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens) # doesn't mess with existing tokens

def generate_dialogue_attention_mask(batch):
  mask = -10000 * torch.ones((batch.shape[0], batch.shape[1], batch.shape[1])) # 12 heads is fixed
  for i in np.arange(batch.shape[0]):
    example_special_idx = torch.nonzero(sum(batch[i] == t for t in (set.union(symbol_dict['SPECIAL_START_TOKEN_IDS'], symbol_dict['SPECIAL_END_TOKEN_IDS'])))).flatten().tolist()
    last_idx = None
    for idx, token_id in enumerate(batch[i].tolist()):
      if token_id == symbol_dict['PAD_TOKEN_ID']:
        break
      if token_id == symbol_dict['BOS_TOKEN_ID'] or token_id == symbol_dict['EOS_TOKEN_ID']:
        mask[i, idx, example_special_idx] = 0 # attend to other special tokens
        mask[i, example_special_idx, idx] = 0 # let other special tokens attend to this
        mask[i, idx, idx] = 0 # attend to self
        if token_id == symbol_dict['EOS_TOKEN_ID']:
          mask[i, idx, 0] = 0 # eos attends to bos
          mask[i, 0, idx] = 0 # bos attends to eos
      elif token_id in symbol_dict['SPECIAL_START_TOKEN_IDS']:
         mask[i, idx, example_special_idx] = 0 # attend to other special tokens
         last_idx = idx
      elif token_id in symbol_dict['SPECIAL_END_TOKEN_IDS']:
         mask[i, idx, example_special_idx] = 0
         span_range = np.arange(last_idx, idx+1) # starts from the last opening special token to including this special token
         x, y = np.meshgrid(span_range, span_range)
         x, y = x.flatten(), y.flatten()
         span_product = np.array(list(zip(x, y))) # 2-D array
         mask[i, span_product[:, 0], span_product[:, 1]] = 0
  # because we use multi-headed attention
  return mask

def train(args, train_dataset, model, tokenizer):
  """ Train the model """
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter()

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  # Check if saved optimizer or scheduler states exist
  if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    os.path.join(args.model_name_or_path, "scheduler.pt")
  ):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
    )

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info(
    "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    args.train_batch_size
    * args.gradient_accumulation_steps
    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
  )
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.exists(args.model_name_or_path):
    # set global_step to gobal_step of last saved checkpoint from model path
    global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    logger.info("  Continuing training from epoch %d", epochs_trained)
    logger.info("  Continuing training from global step %d", global_step)
    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
  )
  set_seed(args)  # Added here for reproductibility
  
  combined_model = "both" in args.task_name
  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):

      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      model.train()
      batch = tuple(t.to(args.device) for t in batch)
      inputs = {
        "input_ids": batch[0], 
        "attention_mask": generate_dialogue_attention_mask(batch[0])
      }
      if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
          batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

      if not combined_model:
        inputs["labels"] = batch[3]
        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
      else:
        outputs = model(**inputs)
        labels = batch[3]
        logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
        m_logits = logits_masked(logits, labels, args.task_name)
        loss = F.cross_entropy(m_logits, labels)

      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps
      
      loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          logs = {}
          if (
            args.local_rank == -1 and args.evaluate_during_training
          ):  # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model, tokenizer, mode="dev")
            for key, value in results.items():
              eval_key = "eval_{}".format(key)
              logs[eval_key] = value

          loss_scalar = (tr_loss - logging_loss) / args.logging_steps
          learning_rate_scalar = scheduler.get_lr()[0]
          logs["learning_rate"] = learning_rate_scalar
          logs["loss"] = loss_scalar
          logging_loss = tr_loss

          for key, value in logs.items():
            tb_writer.add_scalar(key, value, global_step)
          print(json.dumps({**logs, **{"step": global_step}}))

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

          # Save model checkpoint
          output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
          if not os.path.exists(output_dir):
            os.makedirs(output_dir)
          model_to_save = (
            model.module if hasattr(model, "module") else model
          )  # Take care of distributed/parallel training
          model_to_save.save_pretrained(output_dir)
          tokenizer.save_pretrained(output_dir)

          torch.save(args, os.path.join(output_dir, "training_args.bin"))
          logger.info("Saving model checkpoint to %s", output_dir)

          torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
          torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
          logger.info("Saving optimizer and scheduler states to %s", output_dir)

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", mode="dev"):
  eval_task = args.task_name
  eval_output_dir = args.output_dir

  results = {}
  eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, mode)

  if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    os.makedirs(eval_output_dir)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(eval_dataset)
  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu eval
  if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running {} evaluation {} *****".format(mode, prefix))
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)
  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  combined_model = "both" in args.task_name
  for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
      inputs = {
        "input_ids": batch[0], 
        "attention_mask": generate_dialogue_attention_mask(batch[0])
      }
      if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
          batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
      if not combined_model:
        inputs["labels"] = batch[3]
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
      else:
        outputs = model(**inputs)
        labels = batch[3]
        logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
        m_logits = logits_masked(logits, labels, args.task_name)
        tmp_eval_loss = F.cross_entropy(m_logits, labels)
        inputs["labels"] = labels # For the segment of code below
      eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if preds is None:
      preds = logits.detach().cpu().numpy()
      out_label_ids = inputs["labels"].detach().cpu().numpy()
    else:
      preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
      out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

  eval_loss = eval_loss / nb_eval_steps
  if args.output_mode == "classification":
    preds = np.argmax(preds, axis=1)
  elif args.output_mode == "regression":
    preds = np.squeeze(preds)
  result = compute_metrics(preds, out_label_ids, processors[args.task_name].get_labels())
  results.update(result)


  output_eval_file = os.path.join(eval_output_dir, prefix, "{}_results.txt".format(mode))
  with open(output_eval_file, "w") as writer:
    logger.info("***** {} results {} *****".format(mode, prefix))
    for key in sorted(result.keys()):
      logger.info("  %s = %s", key, str(result[key]))
      writer.write("%s = %s\n" % (key, str(result[key])))

  return results

"""
  mode is one of "train", "dev", "test"
"""
def load_and_cache_examples(args, task, tokenizer, mode):
  if args.local_rank not in [-1, 0] and mode == "train":
    torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

  processor = processors[task]
  output_mode = output_modes[task]
  # Load data features from cache or dataset file
  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}_{}_{}".format(
      mode,
      list(filter(None, args.model_name_or_path.split("/"))).pop(),
      str(args.max_seq_length),
      str(task),
    ),
  )
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    examples = (
      processor.get_examples(args.data_dir, mode)
    )
    features = processor.convert_examples_to_features(
      examples,
      tokenizer,
      label_list=label_list,
      max_length=args.max_seq_length,
      output_mode=output_mode,
      pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
      pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save(features, cached_features_file)

  if args.local_rank == 0 and mode == "train":
    torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  if output_mode == "classification":
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
  elif output_mode == "regression":
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

  dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
  return dataset


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    required=True,
    help="The input data dir. Should contain the .json files (or other data files) for the task.",
  )
  parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
  )
  parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
  )
  parser.add_argument(
    "--task_name",
    default=None,
    type=str,
    required=True,
    help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
  )
  parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
  )

  # Other parameters
  parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
  )
  parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
  )
  parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
  )
  parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
  )
  parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
  parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
  parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
  parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
  )
  parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
  )

  parser.add_argument(
    "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
  )
  parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
  )
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
  )
  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
  parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
  )
  parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
  )
  parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

  parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
  parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
  parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
  )
  parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
  parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
  )
  parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
  )
  parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

  parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
  parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
  parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
  args = parser.parse_args()

  if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
  ):
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir
      )
    )

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd

    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )
  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1)
  )

  # Set seed
  set_seed(args)

  # Prepare our task
  args.task_name = args.task_name.lower()
  if args.task_name not in processors:
    raise ValueError("Task not found: %s" % (args.task_name))
  processor = processors[args.task_name]
  args.output_mode = output_modes[args.task_name]
  label_list = processor.get_labels()
  num_labels = len(label_list)

  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  model = model_class.from_pretrained(
    args.model_name_or_path,
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )

  add_special_tokens_(model, tokenizer)
  symbol_dict.update({
    'SPECIAL_START_TOKEN_IDS': set(tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN['additional_special_tokens'])[:3]),
    'SPECIAL_END_TOKEN_IDS': set(tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN['additional_special_tokens'])[3:]),
    'BOS_TOKEN_ID': tokenizer.bos_token_id,
    'EOS_TOKEN_ID': tokenizer.eos_token_id,
    'PAD_TOKEN_ID': tokenizer.pad_token_id
  })

  if args.local_rank == 0:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
  model.to(args.device)

  logger.info("Training/evaluation parameters %s", args)

  # Training
  if args.do_train:
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, "train")
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

  # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
      model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model.to(args.device)

  # Evaluation
  assert not (args.do_test and args.do_eval)
  results = {}
  if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]:
    mode = "dev" if args.do_eval else "test"
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
      checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
      )
      logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate(%s) the following checkpoints: %s", mode, checkpoints)
    for checkpoint in checkpoints:
      logger.info("Checkpoint: %s", checkpoint)
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

      model = model_class.from_pretrained(checkpoint)
      model.to(args.device)
      result = evaluate(args, model, tokenizer, prefix=prefix, mode=mode)
      result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
      results.update(result)
  return results

if __name__ == "__main__":
  main()