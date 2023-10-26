# author: choi sugil
# date: 2023.10.26 version: 1.0.0 license: MIT brief: keyward
# description: pattern reflection (exec)
# Functions can be dynamically created using exec.
import argparse
import sys
import dotenv
import os
import re
import operator
import string


def frequencies_imp(word_list):
    word_freqs = {}
    for w in word_list:
        if w in word_freqs:
            word_freqs[w] += 1
        else:
            word_freqs[w] = 1
    return word_freqs


# if len(sys.argv) > 1:
extract_words_func = "lambda name : [x.lower() for x in re.split(r'[^a-zA-Z]+', open(name).read()) if len(x) > 0 and x.lower() not in stops]"
frequencies_func = "lambda wl : frequencies_imp(wl)"
sort_func = "lambda word_freq : sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)"
filename = "text.txt"
# else:
#     extract_words_func = "lambda x: []"
#     frequencies_func = "lambda x: []"
#     sort_func = "lambda x: []"
#     filename = "text.txt"


# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)

stops = set(open("stop_words.txt").read().split(",") + list(string.ascii_lowercase))


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
    exec("extract_words = " + extract_words_func)
    exec("frequencies = " + frequencies_func)
    exec("sort = " + sort_func)
    word_freqs = locals()["sort"](
        locals()["frequencies"](locals()["extract_words"](args["file"]))
    )
    for w, c in word_freqs[0:25]:
        print(w, " - ", c)


if __name__ == "__main__":
    main()
