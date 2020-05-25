FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data
EVAL=../tools/data_helper.py
UTILS=../tools/utils.py
DETOKENIZER=../tools/mosesdecoder/scripts/tokenizer/detokenizer.perl

NBEST=$3
export CUDA_VISIBLE_DEVICES=1

# $TEXT/id.bpe.test.en
python $FAIRSEQ_ROOT/interactive.py $2 --input ../src.bpe --path $1 \
                                    --nbest $NBEST \
                                    --beam $NBEST --batch-size 256 --buffer-size 2000 --remove-bpe | grep ^H | cut -f3- > temp

perl $DETOKENIZER -l pt < temp > translations.pt
rm temp

python $UTILS --file translations.pt --n $NBEST --remove-duplicate
