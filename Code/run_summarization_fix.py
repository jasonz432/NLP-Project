#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk
import numpy as np
import pandas as pd

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

# Download NLTK punkt tokenizer if not present
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer or not."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use."})
    use_auth_token: bool = field(default=False, metadata={"help": "Use token generated from huggingface-cli login for private models."})
    resize_position_embeddings: Optional[bool] = field(default=None, metadata={"help": "Whether to resize position embeddings if needed."})


@dataclass
class DataTrainingArguments:
    train_file: str = field(metadata={"help": "Path to local training CSV/JSON/JSONL file."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Path to local validation CSV/JSON/JSONL file."})
    test_file: Optional[str] = field(default=None, metadata={"help": "Path to local test CSV/JSON/JSONL file."})
    text_column: str = field(default="text", metadata={"help": "Column name for source texts."})
    summary_column: str = field(default="summary", metadata={"help": "Column name for target summaries."})
    max_source_length: int = field(default=1024, metadata={"help": "Maximum input sequence length after tokenization."})
    max_target_length: int = field(default=128, metadata={"help": "Maximum target sequence length after tokenization."})
    val_max_target_length: Optional[int] = field(default=None, metadata={"help": "Max target length for validation."})
    pad_to_max_length: bool = field(default=False, metadata={"help": "Whether to pad all samples to max length."})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "Number of processes to use for preprocessing."})
    ignore_pad_token_for_loss: bool = field(default=True, metadata={"help": "Ignore padding token in loss computation."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit the number of training samples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit the number of eval samples."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Limit the number of predict samples."})
    num_beams: Optional[int] = field(default=None, metadata={"help": "Number of beams for generation."})
    source_prefix: Optional[str] = field(default="", metadata={"help": "Prefix to add before every source text."})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    set_seed(training_args.seed)

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )

    # Resize embeddings if needed
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    prefix = data_args.source_prefix or ""

    # ------------------- Load Local Files -------------------
    def load_local_file(file_path):
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
            return pd.read_json(file_path, lines=True)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON/JSONL.")

    def preprocess_df(df, text_column, summary_column):
        inputs = [prefix + str(t) for t in df[text_column].tolist()]
        targets = df[summary_column].tolist()
        model_inputs = tokenizer(
            inputs,
            max_length=data_args.max_source_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True
        )
        labels = tokenizer(
            text_target=targets,
            max_length=data_args.max_target_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation=True
        )
        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = eval_dataset = predict_dataset = None
    if training_args.do_train:
        train_df = load_local_file(data_args.train_file)
        if data_args.max_train_samples:
            train_df = train_df.iloc[:data_args.max_train_samples]
        train_dataset = preprocess_df(train_df, data_args.text_column, data_args.summary_column)

    if training_args.do_eval and data_args.validation_file:
        eval_df = load_local_file(data_args.validation_file)
        if data_args.max_eval_samples:
            eval_df = eval_df.iloc[:data_args.max_eval_samples]
        eval_dataset = preprocess_df(eval_df, data_args.text_column, data_args.summary_column)

    if training_args.do_predict and data_args.test_file:
        predict_df = load_local_file(data_args.test_file)
        if data_args.max_predict_samples:
            predict_df = predict_df.iloc[:data_args.max_predict_samples]
        predict_dataset = preprocess_df(predict_df, data_args.text_column, data_args.summary_column)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [p.strip() for p in preds]
        labels = [l.strip() for l in labels]
        preds = ["\n".join(nltk.sent_tokenize(p)) for p in preds]
        labels = ["\n".join(nltk.sent_tokenize(l)) for l in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Prediction
    if training_args.do_predict:
        predict_results = trainer.predict(predict_dataset)
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if training_args.predict_with_generate:
            predictions = tokenizer.batch_decode(
                predict_results.predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))


if __name__ == "__main__":
    main()
