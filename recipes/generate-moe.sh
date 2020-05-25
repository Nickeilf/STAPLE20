FAIRSEQ_ROOT=../tools/fairseq
TEXT=../data
EVAL=../tools/data_helper.py
UTILS=../tools/utils.py
DETOKENIZER=../tools/mosesdecoder/scripts/tokenizer/detokenizer.perl

NBEST=$3
export CUDA_VISIBLE_DEVICES=1


for I in {0..4}
do
    python $FAIRSEQ_ROOT/interactive.py $2 --input $TEXT/id.bpe.test.en --path $1 \
                                    --task translation_moe \
                                    --user-dir $FAIRSEQ_ROOT/examples/translation_moe/src \
                                    --nbest $NBEST \
                                    --method hMoElp --mean-pool-gating-network \
                                    --num-experts 5 \
                                    --gen-expert $I \
                                    --beam $NBEST --batch-size 256 --buffer-size 2000 --remove-bpe | grep ^H | cut -f3- > temp
    perl $DETOKENIZER -l pt < temp > e$I
    rm temp

    python $UTILS --file e$I --n $NBEST --remove-duplicate
done

python $UTILS --combine --file e0,e1,e2,e3,e4
#rm e0 e1 e2 e3 e4



python $EVAL --hyp translations.pt --ref $TEXT/raw/id.test.gold --eval --empty-line-split

