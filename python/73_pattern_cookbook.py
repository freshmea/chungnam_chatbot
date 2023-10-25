#!c:/Python311/python.exe
# 위 shebang 은 'py 73_pattern_1.py' 로 실행할 때만 의미가 있음

# cook book style
import sys
import string

# -*- coding: utf-8 -*-

print(sys.version_info)
print(sys.version)
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


def remove_stop_words():
    global words
    with open("../stop_words.txt") as f:
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
read_file(sys.argv[1])
filter_chars_and_normalize()
scan()
remove_stop_words()
frequencies()
sort()

for tf in word_freqs[0:25]:
    print(tf[0], "-", tf[1])
