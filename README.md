# CBMI: Conditional Bilingual Mutual Information

The implementation of paper “Conditional Bilingual Mutual Information based Adaptive Training for Neural Machine Translation”. 

> This code is based on [fairseq-0.10.2]([pytorch/fairseq: Facebook AI Research Sequence-to-Sequence Toolkit written in Python. (github.com)](https://github.com/pytorch/fairseq)).

## Contents

- [Requirements and Installation](#Requirements and Installation)
- [Datasets](#Datasets)
- [Train](#Train)
- [Test](#Test)

## Requirements and Installation

- PyTorch version >= 1.6
- Python version >= 3.6

## Datasets

- WMT14 En-De
- WMT19 Zh-En

## Train

- WMT14 En-De

  ```shell
  bash train_ende.sh
  ```

- WMT19 Zh-En

  ```shell
  bash train_zhen.sh
  ```

## Test

- WMT14 En-De

  ```shell
  bash test_ende.sh
  ```

- WMT19 Zh-En

  ```shell
  bash test_zhen.sh
  ```

  



