FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data

NAME=id-1pair
OOD_VOCAB=data-bin/shared_bpe40k-ood
OOD_MODEL=checkpoints/ood-naive_enpt/checkpoint30.pt
MODEL=1

SEED=1234
# preprocessing
python $FAIRSEQ_ROOT/preprocess.py --source-lang en --target-lang pt \
                            --trainpref $TEXT/id.bpe.train \
                            --validpref $TEXT/id.bpe.valid \
                            --joined-dictionary \
                            --srcdict $OOD_VOCAB/dict.en.txt \
                            --destdir data-bin/$NAME\
                            --workers 20

# training
export CUDA_VISIBLE_DEVICES=1
python $FAIRSEQ_ROOT/train.py data-bin/$NAME \
    --arch transformer_wmt_en_de \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-epoch 45 \
    --max-tokens 4096 \
    --save-dir checkpoints/id-1pair \
    --restore-file $OOD_MODEL \
    --update-freq 1 \
    --log-interval 50 \
    --fp16 \
    --seed $SEED \
    --num-workers 20 \
    --ddp-backend=c10d \
    --log-format json
