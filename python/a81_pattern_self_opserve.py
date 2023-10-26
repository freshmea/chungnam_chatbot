# author: choi sugil
# date: 2023.10.13 version: 1.0.0 license: MIT brief: keyward
# description: self observe pattern( check caller function name, and locals())
import argparse
import dotenv
import os
import re
import operator
import string
import inspect


def read_stop_words():
    # check caller function name
    if inspect.stack()[1][3] != "extract_words":
        return None
    with open("stop_words.txt") as f:
        stop_words = f.read().split(",")
    stop_words.extend(list(string.ascii_lowercase))
    return stop_words


def extract_words(path_to_file):
    # with open(locals()["path_to_file"]) as f:
    with open(path_to_file) as f:
        str_data = f.read()
    pattern = re.compile(r"[\W_]+")
    word_list = pattern.sub(" ", str_data).lower().split()
    stop_words = read_stop_words()
    return [w for w in word_list if stop_words is None or w not in stop_words]


def frequencies(word_list):
    words_freqs = {}
    for w in locals()["word_list"]:
        if w in words_freqs:
            words_freqs[w] += 1
        else:
            words_freqs[w] = 1
    return words_freqs


def sort(word_freq):
    return sorted(
        locals()["word_freq"].items(), key=operator.itemgetter(1), reverse=True
    )


# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)


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
    word_freqs = sort(frequencies(extract_words(args["file"])))
    for w, c in word_freqs[0:25]:
        print(w, " - ", c)


if __name__ == "__main__":
    main()
