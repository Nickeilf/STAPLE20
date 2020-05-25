FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data

NAME=ood-naive_enpt3
SEED=123

# training
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
python $FAIRSEQ_ROOT/train.py data-bin/shared_bpe40k-ood \
    --arch transformer_wmt_en_de \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 40 \
    --max-tokens 4096 \
    --save-dir checkpoints/$NAME \
    --update-freq 1 \
    --log-interval 500 \
    --fp16 \
    --num-workers 20 \
    --seed $SEED \
    --ddp-backend=c10d \
    --log-format json | tee $NAME.log
