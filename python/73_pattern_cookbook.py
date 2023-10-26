#!c:/Python311/python.exe
# 위 shebang 은 'py 73_pattern_1.py' 로 실행할 때만 의미가 있음

# cook book style
import string
import argparse
import os

CWD = r"C:\chungnam_chatbot\python"
os.chdir(CWD)


# -*- coding: utf-8 -*-
data = []
words = []
word_freqs = []


# procedure
def read_file(path_to_file):
    global data
    with open(path_to_file) as f:
        data = data + list(f.read())


def filter_chars_and_normalize():
    global data
    for i in range(len(data)):
        if not data[i].isalnum():
            data[i] = " "
        else:
            data[i] = data[i].lower()


def scan():
    global data
    global words
    data_str = "".join(data)
    words = words + data_str.split()


def remove_stop_words(file_path):
    global words
    with open(file_path) as f:
        stop_words = f.read().split(",")
    stop_words.extend(list(string.ascii_lowercase))
    indexes = []
    for i in range(len(words)):
        if words[i] in stop_words:
            indexes.append(i)
    for i in reversed(indexes):
        words.pop(i)


def frequencies():
    global words
    global word_freqs
    for w in words:
        word_freqs.append([w, 0])
    for i in range(len(words)):
        for w in word_freqs:
            if w[0] == words[i]:
                w[1] += 1
                break


def sort():
    global word_freqs
    word_freqs.sort(key=lambda x: x[1], reverse=True)


# main
def main():
    # make option parser with default value
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f",
        "--file",
        help="input file",
        default=r"C:\chungnam_chatbot\python\text.txt",
    )
    args = vars(ap.parse_args())
    # print(args.file)

    read_file(args["file"])
    filter_chars_and_normalize()
    scan()
    remove_stop_words(args["file"])
    frequencies()
    sort()

    for tf in word_freqs[0:25]:
        print(tf[0], "-", tf[1])


if __name__ == "__main__":
    main()
