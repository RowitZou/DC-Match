#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

from datasets import load_from_disk
from others.metrics import Metric
from others.data_collator import DataCollator
from models.sm_model import Model
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def train(args):
    print(args)
    if args.task == 'qqp' or args.task == 'medical':
        trainer_args = TrainingArguments(
            output_dir=args.model_path,
            evaluation_strategy="steps",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.test_batch_size,
            gradient_accumulation_steps=args.accum_count,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            max_steps=args.train_steps,
            warmup_steps=0 if not args.warmup else args.warmup_steps,
            logging_steps=args.report_every,
            save_strategy="steps",
            save_steps=args.save_checkpoint_steps,
            eval_steps=args.save_checkpoint_steps,
            no_cuda=True if args.visible_gpus == '-1' else False,
            seed=args.seed,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
    elif args.task == 'mrpc':
        trainer_args = TrainingArguments(
            output_dir=args.model_path,
            evaluation_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.test_batch_size,
            gradient_accumulation_steps=args.accum_count,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            warmup_steps=0,
            logging_steps=args.report_every,
            num_train_epochs=20,
            save_strategy="epoch",
            no_cuda=True if args.visible_gpus == '-1' else False,
            seed=args.seed,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
    dataset = load_from_disk(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    data_collator = DataCollator(args, tokenizer)

    if args.baseline:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=args.num_labels
        )
    else:
        model = Model(args.model, args.num_labels, loss_type=args.loss_type)

    metric = Metric()
    if args.num_labels > 2:
        metric_fct = metric.compute_metrics_macro_f1
    else:
        metric_fct = metric.compute_metrics_f1

    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset['validation'],
        data_collator=data_collator if not args.baseline else None,
        tokenizer=tokenizer,
        compute_metrics=metric_fct
    )

    trainer.train()
    eval_result = trainer.evaluate()
    predict_result = trainer.predict(dataset['test'])
    print(eval_result)
    print(predict_result)
