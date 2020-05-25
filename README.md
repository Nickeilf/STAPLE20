# Exploring Model Consensus to Generate Translation Paraphrases
Training recipes and scripts for system paper "Exploring Model Consensus to Generate Translation Paraphrases" (In WNGT workshop at ACL20) in duolingo [STAPLE](http://sharedtask.duolingo.com/) shared task.

### Setting up
---
You can download the out-of-domain data and required tools by running
``` 
bash prepare_data.sh 
```
If you want to use the in-domain STAPLE dataset, you should download from [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/38OJR6), and extract the training data to `data/raw/`

We trained the model using a modified version of [fairseq](https://github.com/pytorch/fairseq). You may have to compile it by running
```
cd tools/fairseq
pip install --editable .
```

### Preprocessing
---
The preprocessing procedures include
- punctuation normalization, removing non-printable characters
- tokenization
- BPE (shared vocab with size of 40k for both En and Pt)
- parallel data filtering

All data preprocessing are in `data/preprocess_data.sh`

### Training
---
Scripts for pre-training with out-of-domain data and fine-tuning with in-domain data are in the `recipes/` directory. You can also evaluate the weighted marco F1 score with the scripts.

### Citation
---
If you use this work, please cite it as
```
@inproceedings{li-etal-2020-exploring,
  Author    = {Zhenhao Li, and Marina Fomicheva, and Lucia Specia},
  Title     = {Exploring Model Consensus to Generate Translation Paraphrases},
  Booktitle = {Proceedings of the ACL Workshop on Neural Generation and Translation (WNGT)},
  Publisher = {ACL},
  Year      = {2020}
}
```