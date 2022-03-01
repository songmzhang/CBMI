#! /bin/bash
set -e
work_dir=/path/to/your/work/directory
code_dir=$work_dir/fairseq-cbmi
data_dir=/path/to/your/data/wmt19_zhen
bin_dir=$data_dir/data-bin

export PYTHONPATH=$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

name=base_zhen_cbmi
setting=wmt19_zhen/base/cbmi_adaptive
output_dir=$work_dir/ckpts/$setting/$name

if [ ! -d $output_dir ];then
    mkdir -p $output_dir
fi

slang=zh
tlang=en

train(){
    mode=$1
    extra_args=''
    if [[ ${mode} == 'pretrain' ]]; then
        start_to_finetune_steps=9999999
        train_steps=100000
    elif [[ ${mode} == 'finetune' ]]; then
        start_to_finetune_steps=0
        train_steps=200000 
        extra_args='--reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer --fp16'
    elif [[ ${mode} == 'train' ]]; then
        start_to_finetune_steps=100000
        train_steps=300000
    fi
    # run big arch with 'transformer_cbmi_big'
    nohup python $code_dir/fairseq_cli/train.py $bin_dir \
        --task translation --arch transformer_cbmi \
        --share-decoder-input-output-embed \
        --share-lm-decoder-softmax-embed --finetune-fix-lm \
        --source-lang $slang --target-lang $tlang \
        --token-scale 0.1 --sentence-scale 0.3 \
        --pretrain-steps $start_to_finetune_steps --lm-rate 0.01 \
        --dropout 0.1 --weight-decay 0.0 --criterion cbmi_adaptive_label_smoothed_cross_entropy \
        --attention-dropout 0.1 --activation-dropout 0.1 \
        --label-smoothing 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
        --lr 7e-4 --lr-scheduler inverse_sqrt \
        --valid-subset valid,test \
        --eval-bleu --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
        --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --validate-interval-updates 5000 --validate-interval 9999999 \
        --best-checkpoint-metric bleu \
        --maximize-best-checkpoint-metric \
        --warmup-updates 4000 --warmup-init-lr 1e-07 \
        --max-tokens 4096 --update-freq 1 --max-update $train_steps \
        --save-interval-updates 5000 --save-interval 9999999 \
        --keep-interval-updates 10 --keep-best-checkpoints 5 \
        --distributed-world-size 8 --log-interval 100 \
        --num-workers 1 \
        --save-dir $output_dir \
        ${extra_args} \
        >> $output_dir/train.log 2>&1
}

# This script includes validation and test in training, so just grep the corresponding lines in train.log to check the bleu results. To use this, just uncomment the following 'bleu'.
bleu(){
    grep 'INFO | valid' ${output_dir}/train.log | cut -d '|' -f24 > update.num
    grep 'valid: BLEU' ${output_dir}/train.log | cut -d '|' -f4 > valid.results
    grep 'test: BLEU' ${output_dir}/train.log | cut -d '|' -f4 > test.results
    paste -d ' ' update.num valid.results test.results > results
    rm update.num valid.results test.results
}

# Simple 'train' seemingly works a little better than 'pretrain+finetune' on WMT19 Zh-En, same setting on all baselines.
train 'train'
# bleu