FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data

NAME=ood-moe
# preprocessing
# python $FAIRSEQ_ROOT/preprocess.py --source-lang en --target-lang pt \
#                             --trainpref $TEXT/train.clean \
#                             --validpref $TEXT/valid.clean \
#                             --joined-dictionary \
#                             --destdir data-bin/shared_bpe40k-ood \
#                             --workers 20

# training
export CUDA_VISIBLE_DEVICES=1
python $FAIRSEQ_ROOT/train.py data-bin/shared_bpe40k-ood \
    --arch transformer_wmt_en_de \
    --share-all-embeddings \
    --task translation_moe \
    --user-dir $FAIRSEQ_ROOT/examples/translation_moe/src \
    --method hMoElp --mean-pool-gating-network \
    --num-experts 5 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 40 \
    --max-tokens 16384 \
    --save-dir checkpoints/$NAME \
    --update-freq 2 \
    --log-interval 500 \
    --fp16 \
    --num-workers 20 \
    --ddp-backend='no_c10d' \
    --log-format json | tee $NAME-2.log
