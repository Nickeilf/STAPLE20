import os
import argparse
import numpy as np
from random import Random

seed = 1234

""" 
data structure for Duolingo shared task data
"""
class DataSample():
    def __init__(self, src, id):
        self.src = src
        self.id = id
        self.tgts = []
        self.weights = []
        
    def add_tgt(self, tgt):
        self.tgts.append(tgt)
        
    def add_weight(self, weight):
        self.weights.append(weight)
        
def empty_line(sent):
    return len(sent.strip()) == 0

def read_multi_translation_dataset(lines):
    dataset = []

    src_next = True
    for line in lines:
        if src_next:
            line = line.strip().split('|')
            new_data = DataSample(src=line[1], id=line[0])
            dataset.append(new_data)
            src_next = False
        elif empty_line(line):
            src_next = True
        else:
            line = line.strip().split('|')
            dataset[-1].add_tgt(line[0])
            dataset[-1].add_weight(line[1])
    return dataset
    
def extract_data(file, dev_size=500):
    with open(file, 'r') as f:
        data_dir = os.path.dirname(file)
        data_file = f.readlines()
        
    dataset = read_multi_translation_dataset(data_file)

    # set random seed
    Random(seed).shuffle(dataset)
    test = dataset[:dev_size]
    valid = dataset[dev_size:2*dev_size]
    train = dataset[2*dev_size:]


    train_src_file = open(data_dir+'/id.train.en','w') 
    train_tgt_file = open(data_dir+'/id.train.pt','w')
    valid_src_file = open(data_dir+'/id.valid.en', 'w')
    valid_tgt_file = open(data_dir+'/id.valid.pt', 'w')
    test_src_file = open(data_dir+'/id.test.en','w')
    test_tgt_file = open(data_dir+'/id.test.gold','w')
    
    for datasample in valid:
        src = datasample.src
        for translation in datasample.tgts:
            valid_src_file.write(src+"\n")
            valid_tgt_file.write(translation+"\n")

    for datasample in train:
        src = datasample.src
        for translation in datasample.tgts:
            train_src_file.write(src+"\n")
            train_tgt_file.write(translation+"\n")
    
    for datasample in test:
        src = datasample.src
        id = datasample.id
        test_src_file.write(src+"\n")
        test_tgt_file.write(id+"|"+src+"\n")
        for i, translation in enumerate(datasample.tgts):
            weight = datasample.weights[i]
            test_tgt_file.write(translation+"|"+weight+"\n")
        test_tgt_file.write("\n")
            

def extract_data_k(file, dev_size=500, MAX_GOLDS=10):
    with open(file, 'r') as f:
        data_dir = os.path.dirname(file)
        data_file = f.readlines()

    dataset = read_multi_translation_dataset(data_file)

    Random(seed).shuffle(dataset)
    test = dataset[:dev_size]
    valid = dataset[dev_size:2*dev_size]
    train = dataset[2*dev_size:]

    train_src_file = open(data_dir+'/id.train.en', 'w')
    train_tgt_file = open(data_dir+'/id.train.pt', 'w')
    valid_src_file = open(data_dir+'/id.valid.en', 'w')
    valid_tgt_file = open(data_dir+'/id.valid.pt', 'w')
    test_src_file = open(data_dir+'/id.test.en', 'w')
    test_tgt_file = open(data_dir+'/id.test.gold', 'w')

    for datasample in valid:
        src = datasample.src
        for i in range(min(MAX_GOLDS, len(datasample.tgts))):
            translation = datasample.tgts[i]
            valid_src_file.write(src+"\n")
            valid_tgt_file.write(translation+"\n")

    for datasample in train:
        src = datasample.src
        for i in range(min(MAX_GOLDS, len(datasample.tgts))):
            translation = datasample.tgts[i]
            train_src_file.write(src+"\n")
            train_tgt_file.write(translation+"\n")

    for datasample in test:
        src = datasample.src
        id = datasample.id
        test_src_file.write(src+"\n")
        test_tgt_file.write(id+"|"+src+"\n")
        for i, translation in enumerate(datasample.tgts):
            weight = datasample.weights[i]
            test_tgt_file.write(translation+"|"+weight+"\n")
        test_tgt_file.write("\n")

def calculate_F1(datasample, translations):
    WTP = 0
    WFN = 0
    for i, gold_translation in enumerate(datasample.tgts):
        if gold_translation in translations:
            WTP += float(datasample.weights[i])
        else:
            WFN += float(datasample.weights[i])
    recall = WTP / (WTP+WFN)

    precision = 0

    for translation in translations:
        if translation in datasample.tgts:
            precision += 1
    precision /= len(translations)
    
    if precision == 0 and recall == 0:
        F1 = 0
    else:
        F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1

def evaluate(hyp, ref, n):
    with open(ref, 'r') as f:
        ref_file =  f.readlines()
    
    gold_translations = read_multi_translation_dataset(ref_file)

    with open(hyp, 'r') as f:
        hyp_file = f.readlines()
    for i in range(len(hyp_file)):
        hyp_file[i] = hyp_file[i].strip()

    assert n * len(gold_translations) == len(hyp_file)
    
    F1_score = 0
    precision_score = 0
    recall_score = 0
    for i, test_sample in enumerate(gold_translations):
        precision, recall, F1 = calculate_F1(test_sample, hyp_file[i*n:i*n+n])
        precision_score += precision
        recall_score += recall
        F1_score += F1

    precision_score /= len(gold_translations)
    recall_score /= len(gold_translations)
    F1_score /= len(gold_translations)
    print("The weighted marco prec score is: {0:.5f}".format(precision_score))
    print("The weighted marco rec score is: {0:.5f}".format(recall_score))
    print("The weighted marco F1 score is: {0:.5f}".format(F1_score))


def evaluate_line_split(hyp, ref):
    with open(ref, 'r') as f:
        ref_file = f.readlines()
    gold_translations = read_multi_translation_dataset(ref_file)

    with open(hyp, 'r') as f:
        hyp_file = f.readlines()
    for i in range(len(hyp_file)):
        hyp_file[i] = hyp_file[i].strip()

    F1_score = 0
    precision_score = 0
    recall_score = 0

    index = 0
    
    for test_sample in gold_translations:
        temp = []
        while not empty_line(hyp_file[index]):
            temp.append(hyp_file[index])
            index += 1
        index += 1
        precision, recall, F1 = calculate_F1(test_sample, temp)
        precision_score += precision
        recall_score += recall
        F1_score += F1

    precision_score /= len(gold_translations)
    recall_score /= len(gold_translations)
    F1_score /= len(gold_translations)
    print("The weighted marco prec score is: {0:.5f}".format(precision_score))
    print("The weighted marco rec score is: {0:.5f}".format(recall_score))
    print("The weighted marco F1 score is: {0:.5f}".format(F1_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path of the data file")
    parser.add_argument("--hyp", help="path of hypothesis file")
    parser.add_argument("--ref", help="path of reference file")
    parser.add_argument("--extract", default=False, action='store_true', help="Extract file from Duolingo shared task dataset, automatically split train/dev/test")
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--eval", default=False, action='store_true')
    parser.add_argument("--empty-line-split", default=False, action='store_true')
    
    args = parser.parse_args()

    assert args.extract ^ args.eval == True
    
    if args.extract:
        extract_data_k(args.file, MAX_GOLDS=1, dev_size=500)
    elif args.eval and args.empty_line_split:
        evaluate_line_split(args.hyp, args.ref)
    else:
        evaluate(args.hyp, args.ref, args.nbest)
