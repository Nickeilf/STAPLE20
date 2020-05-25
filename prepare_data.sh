# wget https://object.pouta.csc.fi/OPUS-ParaCrawl/v5/moses/en-pt.txt.zip
# unzip en-pt.txt.zip '*.en' '*.pt'
# rm en-pt.txt.zip
# wget https://object.pouta.csc.fi/OPUS-EUbookshop/v2/moses/en-pt.txt.zip
# unzip en-pt.txt.zip '*.en' '*.pt'
# rm en-pt.txt.zip
# wget https://object.pouta.csc.fi/OPUS-Europarl/v8/moses/en-pt.txt.zip
# unzip en-pt.txt.zip '*.en' '*.pt'
# rm en-pt.txt.zip
# wget https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/moses/en-pt.txt.zip
# unzip en-pt.txt.zip '*.en' '*.pt'
# rm en-pt.txt.zip
wget https://object.pouta.csc.fi/OPUS-QED/v2.0a/moses/en-pt.txt.zip
unzip en-pt.txt.zip '*.en' '*.pt'
rm en-pt.txt.zip
wget https://object.pouta.csc.fi/OPUS-Tatoeba/v20190709/moses/en-pt.txt.zip
unzip en-pt.txt.zip '*.en' '*.pt'
rm en-pt.txt.zip



mkdir data/raw
cat *.en > data/raw/ood.en
cat *.pt > data/raw/ood.pt

rm *.en
rm *.pt

git clone https://github.com/rsennrich/subword-nmt.git tools/subword-nmt
git clone https://github.com/moses-smt/mosesdecoder.git tools/mosesdecoder