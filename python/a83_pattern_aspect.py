# author: choi sugil
# date: 2023.10.13 version: 1.0.0 license: MIT brief: keyward
# description: aspect pattern
import argparse
import dotenv
import os
import re
import operator
import string
import time


# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)


def extract_words(path_to_file):
    with open(path_to_file) as f:
        str_data = f.read()
    pattern = re.compile(r"[\W_]+")
    word_list = pattern.sub(" ", str_data).lower().split()
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


def profile(f):
    def profilewrapper(*arg, **kw):
        start_time = time.time()
        ret_value = f(*arg, **kw)
        elapsed = time.time() - start_time
        print(f"{f.__name__}(...) took {elapsed:.10f} secs")
        return ret_value

    return profilewrapper


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
    tracked_functions = [extract_words, frequencies, sort]
    for func in tracked_functions:
        # The function is being redefined using a profile wrapper here.
        globals()[func.__name__] = profile(func)

    word_freqs = sort(frequencies(extract_words(args["file"])))

    for w, c in word_freqs[0:25]:
        print(f"{w} - {c}")


if __name__ == "__main__":
    main()
