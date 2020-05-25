import os
import argparse
import data_helper as data
from collections import Counter

def duplicate(file, N, out_file):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    result = [line for line in lines for i in range(N)]
    
    with open(out_file, 'w') as f:
        f.writelines(result)

# remove duplicate and add empty line as delimiter
def remove_duplicate(file, N):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    result = []
    temp = []

    for i, line in enumerate(lines):
        if line not in temp:
            temp.append(line)
            result.append(line)
        if i % N == N - 1:
            result.append("\n")
            temp = []
            
    with open(file, 'w') as f:
        f.writelines(result)


def split_test_data(file):
    with open(file,'r') as f:
        lines = f.readlines()

    id = []
    src = []

    for line in lines:
        temp = line.split("|")
        id.append(temp[0]+"\n")
        src.append(temp[1])
    
    with open("id.txt", "w") as f:
        f.writelines(id)
    with open("src.txt", "w") as f:
        f.writelines(src)

def merge_translation(file, hyp):
    with open(file, 'r') as f:
        ids = f.readlines()
    with open(hyp, 'r') as f:
        translations = f.readlines()

    result = []
    index = 0
    translation_next = True

    temp = []
    for line in translations:
        if not data.empty_line(line):
            temp.append(line)
        else:
            result.append(ids[index])
            index += 1
            result += temp
            temp = []
            result += "\n"
    
    with open("en_pt", "w") as f:
        f.writelines(result)

# combine several translation files into one (for ensemble method use)
def combine_files(files):
    file_names = files.split(',')
    
    translations = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            translations.append(f.readlines())
    
    translations_grouped_list = []
    for translation_set in translations:
        temp = "&&&".join(translation_set)
        temp = temp.split("&&&\n&&&")
        temp[-1] = temp[-1][:-4]
        translations_grouped_list.append(temp)

    length = len(translations_grouped_list[0])
    
    
    result = []
    for i in range(length):
        temp = []
        for model_translations in translations_grouped_list:
            temp += model_translations[i].split("&&&")
        result += list(set(temp))
        result += "\n"
        
    with open("translations.pt", 'w') as f:
        f.writelines(result)

def rerank_files(files, threshold=1):
    file_names = files.split(',')

    translations = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            translations.append(f.readlines())

    translations_grouped_list = []
    for translation_set in translations:
        temp = "&&&".join(translation_set)
        temp = temp.split("&&&\n&&&")
        temp[-1] = temp[-1][:-4]
        translations_grouped_list.append(temp)

    length = len(translations_grouped_list[0])

    result = []
    for i in range(length):
        temp = []
        for model_translations in translations_grouped_list:
            temp += model_translations[i].split("&&&")
        count = Counter(temp)
        temp = [key for key, value in count.items() if value > threshold]
        result += temp
        result += "\n"

    with open("translations.pt", 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path of the data file")
    parser.add_argument("--hyp", help="path of the hypothesis file")
    parser.add_argument("--out_file", help="path of the output file")
    parser.add_argument("--remove-duplicate", action='store_true', default=False)
    parser.add_argument("--split", action='store_true', default=False)
    parser.add_argument("--merge", action='store_true', default=False)
    parser.add_argument("--combine", action='store_true', default=False)
    parser.add_argument("--rerank", action='store_true', default=False)
    parser.add_argument("--n", type=int, default=1)

    args = parser.parse_args()

    if args.remove_duplicate:
        remove_duplicate(args.file, args.n)
    elif args.split:
        split_test_data(args.file)
    elif args.rerank:
        rerank_files(args.file, args.n)
    elif args.merge:
        merge_translation(args.file, args.hyp)
    elif args.combine:
        combine_files(args.file)
    else:
        duplicate(args.file, args.n, args.out_file)
    
