#!c:/Python311/python.exe
# author: choi sugil
# date: 2023.10.26 version: 1.0.0 license: MIT brief: keyward
# description: pattern object style
from abc import ABCMeta
import argparse
import operator
import re
import string
import dotenv
import os
import time
import functools


# runtime check decorator
def iter_profile(iter=1):
    def profile(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            time_list = []
            result = None
            for i in range(iter):
                start_time = time.time()
                print("*" * 10 + f" {i+1}th iteration start " + "*" * 10)
                result = func(*args, **kwargs)
                end_time = time.time()
                time_list.append(end_time - start_time)
            print(
                f"func: {func.__name__}, elapsed time: {sum(time_list)/len(time_list)}"
            )
            return result

        return wrapper

    return profile


# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)


class TFExercise:
    __metaclass__ = ABCMeta

    def info(self):
        return self.__class__.__name__


class DataStorageManager(TFExercise):
    def __init__(self, path_to_file):
        with open(path_to_file) as f:
            self.data = f.read()
        pattern = re.compile(r"[\W_]+")
        self.data = pattern.sub(" ", self.data).lower()

    def words(self):
        return self.data.split()

    def info(self):
        return (
            super(DataStorageManager, self).info()
            + ": My major data structure is a "
            + self.data.__class__.__name__
        )


class StopWordManager(TFExercise):
    def __init__(self):
        with open("stop_words.txt") as f:
            self.stop_words = f.read().split(",")
        self.stop_words.extend(list(string.ascii_lowercase))

    def is_stop_word(self, word):
        return word in self.stop_words

    def info(self):
        return (
            super(StopWordManager, self).info()
            + ": My major data structure is a "
            + self.stop_words.__class__.__name__
        )


class WordFrequencyManager(TFExercise):
    def __init__(self):
        self.word_freqs = {}

    def increment_count(self, word):
        if word in self.word_freqs:
            self.word_freqs[word] += 1
        else:
            self.word_freqs[word] = 1

    def sorted(self):
        return sorted(self.word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    def info(self):
        return (
            super(WordFrequencyManager, self).info()
            + ": My major data structure is a "
            + self.word_freqs.__class__.__name__
        )


class WordFrequencyController(TFExercise):
    def __init__(self, path_to_file):
        self.storage_manager = DataStorageManager(path_to_file)
        self.stop_word_manager = StopWordManager()
        self.word_freq_manager = WordFrequencyManager()

    def run(self):
        for w in self.storage_manager.words():
            if not self.stop_word_manager.is_stop_word(w):
                self.word_freq_manager.increment_count(w)

        word_freqs = self.word_freq_manager.sorted()
        for w, c in word_freqs[0:25]:
            print(w, "-", c)


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
    # calculate run time by using decorator
    WordFrequencyController(args["file"]).run()


if __name__ == "__main__":
    main()
