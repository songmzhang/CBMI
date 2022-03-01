# Conditional Bilingual Mutual Information based Adaptive Training for Neural Machine Translation

The implementation of ACL2022 paper “Conditional Bilingual Mutual Information based Adaptive Training for Neural Machine Translation”. 

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

  



