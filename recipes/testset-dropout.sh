FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data
EVAL=../tools/data_helper.py
UTILS=../tools/utils.py
DETOKENIZER=../tools/mosesdecoder/scripts/tokenizer/detokenizer.perl

NBEST=$3
NUM_DUPLICATE=$4
export CUDA_VISIBLE_DEVICES=1

python $UTILS --file ../test.src.bpe --n $NUM_DUPLICATE --out_file test.bpe.dup

python $FAIRSEQ_ROOT/interactive.py $2 --input test.bpe.dup --path $1 \
                                    --nbest $NBEST --retain-dropout \
                                    --beam $NBEST --batch-size 256 --buffer-size 2000 --remove-bpe | grep ^H | cut -f3- > temp

perl $DETOKENIZER -l pt < temp > translations.pt
rm temp
rm test.bpe.dup

python $UTILS --file translations.pt --n $(($3 * $4)) --remove-duplicate

# python $EVAL --hyp translations.pt --ref $TEXT/raw/id.test.gold --eval --empty-line-split
