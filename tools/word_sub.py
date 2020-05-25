import os
import argparse
from collections import Counter
from tqdm import tqdm
from data_helper import read_multi_translation_dataset
import re

def tokenize(str):
    return re.findall("[.,?!]|[óâáéºãªàõíêôú\w$:\-\+]+", str)

def learn_word_substitutions(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        
    with open("stop2.txt", "r") as f:
        stopwords = f.readlines()
    stopwords = [stopword.strip() for stopword in stopwords]
        
    word_sub_counter = Counter()
    
    temp = [line.strip() for line in lines if len(line.strip()) > 0 ]
    corpus = []
    for sentence in temp:
        corpus += tokenize(sentence)
    word_counter = Counter(corpus)
    
    
    temp = []
    for line in tqdm(lines):
        if len(line.strip()) > 0:
            temp.append(line.strip())
        else:
            count_word_sub(temp, word_sub_counter, stopwords)
            temp = []
    
    word_sub_prob = Counter()
    for key, value in word_sub_counter.items():
        w1 = key.split(" == ")[0]
        prob = value / word_counter[w1]
        word_sub_prob[key] = prob if prob < 1 else 1
    
    sorted_counter = sorted(word_sub_counter.items(), key=lambda kv: kv[1], reverse=True)
    sorted_prob_counter = sorted(word_sub_prob.items(), key=lambda kv: kv[1], reverse=True)
    
    output_file = open("sub_freq.txt", "w")
    for key,value in sorted_counter:
        words = key.split(" == ")
        reverse = words[1] + " == " + words[0]
        if value != 0 and word_sub_counter[reverse] != 0:
            word_sub_counter[reverse] = 0
        if value != 0:
            output_file.write(words[0]+" == "+words[1]+"\t" + str(value) + "\n")
            
    output_file = open("sub_prob.txt", "w")
    for key, value in sorted_prob_counter:
        words = key.split(" == ")
        reverse = words[1] + " == " + words[0]
        if value != 0:
            output_file.write(words[0]+" == "+words[1] +
                              "\t" + str(value) + "\n")
            
def count_word_sub(sent_list, counter, stopwords):
    for s1 in sent_list:
        for s2 in sent_list:
            if s1 == s2:
                continue
            else:
                s1_word = tokenize(s1)
                s2_word = tokenize(s2)
                
                if len(s1_word) == len(s2_word):
                    count = 0
                    for i in range(len(s1_word)):
                        if s1_word[i] == s2_word[i]:
                            continue
                        else:
                            count += 1
                            w1 = s1_word[i]
                            w2 = s2_word[i]
                            if count > 1:
                                break
                    if count == 1 and w1 not in stopwords and w2 not in stopwords:
                        counter[w1+" == "+w2] += 1
                        
def apply_word_substitutions(file, dict, T, W=9):
    with open(dict, "r") as f:
        substitutions = f.readlines()
    
    subs = []
    # filter substitutions based on threshold
    for line in substitutions:
        element = line.strip().split("\t")
        sub = element[0].split(" == ")
        count = float(element[1])
        
        if count > T:
            subs.append((sub[0], sub[1]))
    print(len(subs))
    # add substitutions to file
    with open(file, "r") as f:
        hyp_file = f.readlines()
    
    translations_set = "&&&".join(hyp_file)
    translations_set = translations_set.split("&&&\n&&&")
    translations_set[-1] = translations_set[-1][:-4]
    
    translations_set = [translations.split("&&&") for translations in translations_set]
    
    result = []
    for i, translations in enumerate(tqdm(translations_set)):
        translations = [sent.strip() for sent in translations]
        temp = []
        for sentence in translations:
            for k, word in enumerate(tokenize(sentence)):
                for (w1, w2) in subs:
                    if word == w1:
                        subed_sentence = get_subed_sent(sentence, k, w2)
                        translations_set[i].append(subed_sentence)
                        temp.append(subed_sentence)
        temp = list(set(temp))
        result.append(temp)
        translations_set[i] = list(set(translations_set[i]))
        
    with open("subed_translations.pt", "w") as f:
        for translations in translations_set:
            if len(translations) == 0:
                f.write("EMPTYLINE\n")
            else:
                for translation in translations:
                    for _ in range(W):
                        f.write(translation)
            f.write("\n")
        
def get_subed_sent(sentence, idx, w2):
    words = tokenize(sentence)
    words[idx] = w2
    for i, word in enumerate(words):
        if word in ",.!?":
            words[i-1] = words[i-1] + word
            del words[i]
    return " ".join(words)  + "\n"


def extract_data_ref(file):
    with open(file, 'r') as f:
        data_dir = os.path.dirname(file)
        data_file = f.readlines()

    dataset = read_multi_translation_dataset(data_file)

    output_file = open("references", 'w')

    for datasample in dataset:
        src = datasample.src
        for translation in datasample.tgts:
            output_file.write(translation+"\n")
        output_file.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path of the task training data file from STAPLE dataset")
    parser.add_argument("--dict", help="path of the dict file")
    parser.add_argument("--learn", action='store_true', default=False)
    parser.add_argument("--apply", action='store_true', default=False)
    parser.add_argument("--threshold", type=float, default=500)

    args = parser.parse_args()

    if args.learn:
        if not os.path.isfile('references'):
            extract_data_ref(args.file)
        learn_word_substitutions("references")
    else:
        apply_word_substitutions(args.file, args.dict, args.threshold)
