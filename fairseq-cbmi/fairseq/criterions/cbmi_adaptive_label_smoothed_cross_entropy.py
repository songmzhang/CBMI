# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    lm_label_smoothing: float = field(
        default=0.1,
        metadata={"help": "epsilon for language model label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    token_scale: float = field(
        default=0.0,
        metadata={"help": "hyperparameter for token cbmi"},
    )
    sentence_scale: float = field(
        default=0.0,
        metadata={"help": "hyperparameter for sentence cbmi"},
    )
    pretrain_steps: int = field(
        default=100000,
        metadata={"help": "step for ending pretrain and starting finetune"},
    )
    lm_rate: float = field(
        default=0.01,
        metadata={"help": "lm loss rate"},
    )
    finetune_fix_lm: bool = field(
        default=False,
        metadata={"help": "fix language model when finetuning"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "cbmi_adaptive_label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class CBMIAdaptiveLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        lm_label_smoothing=0.1,
        token_scale=0.0,
        sentence_scale=0.0,
        pretrain_steps=100000,
        lm_rate=0.01,
        finetune_fix_lm=False,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.lm_eps = lm_label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.token_scale = token_scale
        self.sentence_scale = sentence_scale
        self.pretrain_steps = pretrain_steps
        self.lm_rate = lm_rate
        self.finetune_fix_lm=finetune_fix_lm

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        nmt_loss, nmt_nll_loss, lm_loss, lm_nll_loss, log = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": nmt_loss.data,
            "nll_loss": nmt_nll_loss.data,
            "lm_loss": lm_loss.data,
            "lm_nll_loss": lm_nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        for key in log:
            logging_output[key] = log[key]
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if self.finetune_fix_lm and model.num_updates > self.pretrain_steps:
            lm_loss = lm_loss.detach()

        loss = nmt_loss + self.lm_rate * lm_loss
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        # Rather than using ori api, rewrite cbmi calculation here
        # =========== CBMI-based adaptive loss =========== #
        nmt_output, lm_output = net_output
        nmt_logits = nmt_output[0]
        lm_logits = lm_output[0]
        nmt_probs = utils.softmax(nmt_logits, -1).reshape(-1, nmt_logits.shape[-1])
        lm_probs = utils.softmax(lm_logits, -1).reshape(-1, lm_logits.shape[-1])
        nmt_lprobs = torch.log(nmt_probs)
        lm_lprobs = torch.log(lm_probs)
        
        target = sample["target"]
        pad_mask = target.ne(self.padding_idx)
        shape = target.shape
        target = target.reshape(-1)
        if target.dim() == nmt_logits.dim() - 1:
            target = target.unsqueeze(-1)

        nmt_loss, nmt_nll_loss = label_smoothed_nll_loss(
            nmt_lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=False,
        )
        lm_loss, lm_nll_loss = label_smoothed_nll_loss(
            lm_lprobs,
            target,
            self.lm_eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        num_updates = model.num_updates
        if num_updates > self.pretrain_steps:
            cbmi = torch.log(nmt_probs / (lm_probs + 1e-9))    # in case that lm_probs are too little
            cbmi = cbmi.detach()
            golden_cbmi = torch.gather(cbmi, -1, index=target.unsqueeze(-1))
            # token-weight
            token_cbmi = golden_cbmi.reshape(shape)
            mean_token_cbmi = (token_cbmi * pad_mask).sum(-1, keepdims=True) / pad_mask.sum(-1, keepdims=True)
            std_token_cbmi = torch.sqrt(torch.sum((token_cbmi - mean_token_cbmi) ** 2 * pad_mask, -1, keepdims=True) / pad_mask.shape[-1])
            norm_token_cbmi = (token_cbmi - mean_token_cbmi) / std_token_cbmi
            token_weight = torch.where(self.token_scale * norm_token_cbmi + 1.0 >= 0, 
                                       self.token_scale * norm_token_cbmi + 1.0, 
                                       torch.zeros_like(norm_token_cbmi))
            # sentence-weight
            sentence_cbmi = mean_token_cbmi
            mean_sentence_cbmi = sentence_cbmi.mean(0, keepdims=True)
            std_sentence_cbmi = torch.sqrt(torch.sum((sentence_cbmi - mean_sentence_cbmi) ** 2, 0, keepdims=True) / pad_mask.shape[-1])
            norm_sentence_cbmi = (sentence_cbmi - mean_sentence_cbmi) / std_sentence_cbmi
            sentence_weight = torch.where(self.sentence_scale * norm_sentence_cbmi + 1.0 >= 0, 
                                          self.sentence_scale * norm_sentence_cbmi + 1.0, 
                                          torch.zeros_like(norm_sentence_cbmi))
            # final-weight
            weight = token_weight * sentence_weight
            weight = weight.detach()
            nmt_loss = nmt_loss.reshape(shape)
            nmt_loss = weight * nmt_loss 
            # logging output
            mean_cbmi = (token_cbmi * pad_mask).sum() / pad_mask.sum()
            std_cbmi = torch.sqrt(((token_cbmi - mean_cbmi) ** 2 * pad_mask).sum() / pad_mask.sum())
            max_weight = weight.max()
            min_weight = weight.min()
            zero_rate = torch.div((weight.eq(0) * pad_mask).sum(), pad_mask.sum())
        else:
            mean_cbmi = 0.0
            std_cbmi = 0.0
            max_weight = 0.0
            min_weight = 0.0
            zero_rate = 0.0
            
        logging_output = {
            "mean_cbmi": mean_cbmi, 
            "std_cbmi": std_cbmi, 
            "max_weight": max_weight, 
            "min_weight": min_weight, 
            "zero_rate": zero_rate,
        }
        if reduce:
            nmt_loss = nmt_loss.sum()
            nmt_nll_loss = nmt_nll_loss.sum()

        return nmt_loss, nmt_nll_loss, lm_loss, lm_nll_loss, logging_output

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        gpu_num = len(logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        lm_loss_sum = sum(log.get("lm_loss", 0) for log in logging_outputs)
        lm_nll_loss_sum = sum(log.get("lm_nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mean_cbmi_sum = sum(log.get("mean_cbmi", 0) for log in logging_outputs)
        std_cbmi_sum = sum(log.get("std_cbmi", 0) for log in logging_outputs)
        max_weight_sum = sum(log.get("max_weight", 0) for log in logging_outputs)
        min_weight_sum = sum(log.get("min_weight", 0) for log in logging_outputs)
        zero_rate_sum = sum(log.get("zero_rate", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "lm_loss", lm_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "lm_nll_loss", lm_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        # log cbmi information
        metrics.log_scalar(
            "mean_cbmi", mean_cbmi_sum / 8 , ntokens, round=3
        )
        metrics.log_scalar(
            "std_cbmi", std_cbmi_sum / 8, ntokens, round=3
        )
        metrics.log_scalar(
            "max_weight", max_weight_sum / 8, ntokens, round=3
        )
        metrics.log_scalar(
            "min_weight", min_weight_sum / 8, ntokens, round=3
        )
        metrics.log_scalar(
            "zero_rate", zero_rate_sum / 8, ntokens, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
