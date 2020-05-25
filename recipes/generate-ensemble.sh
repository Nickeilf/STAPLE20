FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data
EVAL=../tools/data_helper.py
UTILS=../tools/utils.py
DETOKENIZER=../tools/mosesdecoder/scripts/tokenizer/detokenizer.perl


NFILES=$1

python $UTILS --combine --file trans1,trans2,trans3
python $EVAL --hyp translations.pt --ref $TEXT/raw/id.test.gold --eval --empty-line-split
