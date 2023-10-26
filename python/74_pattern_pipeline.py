#!c:/Python311/python.exe
# author: choi sugil
# date: 2023.10.26 version: 1.0.0 license: MIT brief: keyward
# description: pattern pipeline style
import argparse
import operator
import re
import string
import dotenv
import os
import time


# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)

# function


def read_file(path_to_file):
    with open(path_to_file) as f:
        data = f.read()
    return data


def filter_chars_and_normalize(str_data):
    pattern = re.compile(r"[\W_]+")
    return pattern.sub(" ", str_data).lower()


def scan(str_data):
    return str_data.split()


def remove_stop_words(word_list):
    with open("stop_words.txt") as f:
        stop_words = f.read().split(",")
    stop_words.extend(list(string.ascii_lowercase))
    return [w for w in word_list if w not in stop_words]


def frequencies(word_list):
    word_freqs = {}
    for w in word_list:
        if w in word_freqs:
            word_freqs[w] += 1
        else:
            word_freqs[w] = 1
    return word_freqs


def sort(word_freq):
    return sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)


def print_all(word_freqs):
    if len(word_freqs) > 0:
        print(word_freqs[0][0], "-", word_freqs[0][1])
        print_all(word_freqs[1:])


# main
def main():
    # make option parser with default value
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f",
        "--file",
        help="input file",
        default="text.txt",
    )
    args = vars(ap.parse_args())

    start_time = time.time()
    print_all(
        sort(
            frequencies(
                remove_stop_words(
                    scan(filter_chars_and_normalize(read_file(args["file"])))
                )
            )
        )[:25]
    )
    end_time = time.time()
    print(f"func: {__name__}, elapsed time: {end_time - start_time}")


if __name__ == "__main__":
    main()
