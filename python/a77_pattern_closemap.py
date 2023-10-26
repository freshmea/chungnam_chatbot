# author: choi sugil
# date: 2023.10.13 version: 1.0.0 license: MIT brief: keyward
# description:
import argparse
import re
import dotenv
import os
import string
import operator
from a75_pattern_object import iter_profile

# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)


def extract_words(obj, path_to_file):
    with open(path_to_file) as f:
        obj["data"] = f.read()
    pattern = re.compile(r"[\W_]+")
    data_str = pattern.sub(" ", obj["data"]).lower()
    obj["data"] = data_str.split()


def load_stop_words(obj):
    with open("stop_words.txt") as f:
        obj["stop_words"] = f.read().split(",")
    obj["stop_words"].extend(list(string.ascii_lowercase))


def increment_count(obj, w):
    obj["freqs"][w] = 1 if w not in obj["freqs"] else obj["freqs"][w] + 1


@iter_profile()
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
    data_storage_obj = {
        "data": [],
        "init": lambda path_to_file: extract_words(data_storage_obj, path_to_file),
        "words": lambda: data_storage_obj["data"],
    }

    stop_words_obj = {
        "stop_words": [],
        "init": lambda: load_stop_words(stop_words_obj),
        "is_stop_word": lambda word: word in stop_words_obj["stop_words"],
    }

    word_freqs_obj = {
        "freqs": {},
        "increment_count": lambda w: increment_count(word_freqs_obj, w),
        "sorted": lambda: sorted(
            word_freqs_obj["freqs"].items(), key=operator.itemgetter(1), reverse=True
        ),
    }

    data_storage_obj["init"](args["file"])
    stop_words_obj["init"]()

    for w in data_storage_obj["words"]():
        if not stop_words_obj["is_stop_word"](w):
            word_freqs_obj["increment_count"](w)

    word_freqs = word_freqs_obj["sorted"]()
    for w, c in word_freqs[0:25]:
        print(w, "-", c)


if __name__ == "__main__":
    main()
