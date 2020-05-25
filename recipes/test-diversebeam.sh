FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data
EVAL=../tools/data_helper.py
UTILS=../tools/utils.py
DETOKENIZER=../tools/mosesdecoder/scripts/tokenizer/detokenizer.perl

NBEST=$3
BEAM=$4
STRENGTH=$5
export CUDA_VISIBLE_DEVICES=1

python $FAIRSEQ_ROOT/interactive.py $2 --input ../src.bpe --path $1 \
                                    --nbest $NBEST --diverse-beam-groups $BEAM --diverse-beam-strength $STRENGTH --beam $BEAM \
                                    --batch-size 256 --buffer-size 2000 --remove-bpe | grep ^H | cut -f3- > temp

perl $DETOKENIZER -l pt < temp > translations.pt
rm temp

python $UTILS --file translations.pt --n $3 --remove-duplicate

#python $EVAL --hyp translations.pt --ref $TEXT/raw/id.test.gold --eval --empty-line-split
