SCRIPTS=../tools/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
TRUECASER=$SCRIPTS/recaser
BPEROOT=../tools/subword-nmt/subword_nmt

RAW_DATA=raw
src=en
tgt=pt

mkdir bpe tok

# tokenize
# for l in $src $tgt ; do
#     cat $RAW_DATA/ood.$l | \
#         perl $NORM_PUNC $l | \
#         perl $REM_NON_PRINT_CHAR | \
#         perl $TOKENIZER -thread 20 -a -l $l > tok/ood.tok.$l

#     perl $TRUECASER/train-truecaser.perl --model tok/truecase-model.$l --corpus tok/ood.tok.$l
#     perl $TRUECASER/truecase.perl --model tok/truecase-model.$l < tok/ood.tok.$l > tok/ood.true.$l
# done

python ../tools/data_helper.py --file raw/train.en_pt.2020-01-13.gold.txt --extract
perl $TOKENIZER -thread 20 -a -l en < raw/id.train.en > tok/id.tok.train.en
perl $TOKENIZER -thread 20 -a -l en < raw/id.valid.en > tok/id.tok.valid.en
perl $TOKENIZER -thread 20 -a -l en < raw/id.test.en > tok/id.tok.test.en
perl $TOKENIZER -thread 20 -a -l pt < raw/id.train.pt > tok/id.tok.train.pt
perl $TOKENIZER -thread 20 -a -l pt < raw/id.valid.pt > tok/id.tok.valid.pt

# split train/valid/test
# echo "splitting train and valid..."
# for l in $src $tgt; do
#     awk '{if (NR%10000 == 0)  print $0; }' tok/ood.true.$l > tok/test.$l
#     awk '{if (NR%10000 != 0)  print $0; }' tok/ood.true.$l > tok/train_valid.$l
#     awk '{if (NR%10000 == 0)  print $0; }' tok/train_valid.$l > tok/valid.$l
#     awk '{if (NR%10000 != 0)  print $0; }' tok/train_valid.$l > tok/train.$l
#     rm tok/train_valid.*
# done

# byte pair encoding
TRAIN=tok/train.$src-$tgt
BPE_CODE=bpe/bpe.combined
# BPE_TOKENS=40000
# for l in $src $tgt; do
#     cat tok/train.$l >> $TRAIN
# done
# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# for L in $src $tgt; do
#     for f in tok/train.$L tok/valid.$L tok/test.$L; do
#         echo "apply_bpe.py to ${f}..."
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > bpe/bpe.$f
#     done
# done

python $BPEROOT/apply_bpe.py -c $BPE_CODE < tok/id.tok.train.en > id.bpe.train.en
python $BPEROOT/apply_bpe.py -c $BPE_CODE < tok/id.tok.train.pt > id.bpe.train.pt
python $BPEROOT/apply_bpe.py -c $BPE_CODE < tok/id.tok.valid.en > id.bpe.valid.en
python $BPEROOT/apply_bpe.py -c $BPE_CODE < tok/id.tok.valid.pt > id.bpe.valid.pt
python $BPEROOT/apply_bpe.py -c $BPE_CODE < tok/id.tok.test.en > id.bpe.test.en

python ../tools/shuffle.py -src id.bpe.train.en -tgt id.bpe.train.pt

# perl $CLEAN -ratio 1.5 bpe/bpe.train $src $tgt ./train.clean 1 250
# perl $CLEAN -ratio 1.5 bpe/bpe.valid $src $tgt ./valid.clean 1 250

# cp bpe/bpe.test.* ./