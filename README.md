# Conditional Bilingual Mutual Information based Adaptive Training for Neural Machine Translation

The implementation of ACL2022 paper “Conditional Bilingual Mutual Information based Adaptive Training for Neural Machine Translation”. [[paper]](https://arxiv.org/abs/2203.02951)

> This code is based on [fairseq-0.10.2](https://github.com/pytorch/fairseq).

## Contents

- [Requirements and Installation](#Requirements-and-Installation)
- [Datasets](#Datasets)
- [Train](#Train)
- [Test](#Test)

## Requirements and Installation

- PyTorch version >= 1.6
- Python version >= 3.6
- sacremoses version == 0.0.47
- fairseq version == 0.10.2

## Datasets

- WMT14 En-De (4.5M)
- WMT19 Zh-En (20M)

## Prepare

Before starting to run the scripts, please build fairseq-cbmi first:
```shell
pip install --editable ./
```

## Train

- WMT14 En-De

  ```shell
  bash train_ende.sh &
  ```

- WMT19 Zh-En

  ```shell
  bash train_zhen.sh &
  ```

## Test

These scripts conduct validation and test during the training process. You can directly check the **BLEU** results through uncommenting the 'bleu' function line in the corresponding train scripts.

## Citation
Please cite this paper if you find this repo useful.
```
@inproceedings{zhang-etal-2022-conditional,
    title = "Conditional Bilingual Mutual Information Based Adaptive Training for Neural Machine Translation",
    author = "Zhang, Songming  and
      Liu, Yijin  and
      Meng, Fandong  and
      Chen, Yufeng  and
      Xu, Jinan  and
      Liu, Jian  and
      Zhou, Jie",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.169",
    doi = "10.18653/v1/2022.acl-long.169",
    pages = "2377--2389",
    abstract = "Token-level adaptive training approaches can alleviate the token imbalance problem and thus improve neural machine translation, through re-weighting the losses of different target tokens based on specific statistical metrics (e.g., token frequency or mutual information). Given that standard translation models make predictions on the condition of previous target contexts, we argue that the above statistical metrics ignore target context information and may assign inappropriate weights to target tokens. While one possible solution is to directly take target contexts into these statistical metrics, the target-context-aware statistical computing is extremely expensive, and the corresponding storage overhead is unrealistic. To solve the above issues, we propose a target-context-aware metric, named conditional bilingual mutual information (CBMI), which makes it feasible to supplement target context information for statistical metrics. Particularly, our CBMI can be formalized as the log quotient of the translation model probability and language model probability by decomposing the conditional joint distribution. Thus CBMI can be efficiently calculated during model training without any pre-specific statistical calculations and large storage overhead. Furthermore, we propose an effective adaptive training approach based on both the token- and sentence-level CBMI. Experimental results on WMT14 English-German and WMT19 Chinese-English tasks show our approach can significantly outperform the Transformer baseline and other related methods.",
}
```
  



