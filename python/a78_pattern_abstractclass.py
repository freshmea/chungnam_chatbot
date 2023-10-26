# author: choi sugil
# date: 2023.10.13 version: 1.0.0 license: MIT brief: keyward
# description: abstract class pattern
import argparse
import dotenv
import os
import abc
import re
import operator
import string
from a75_pattern_object import iter_profile

# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)


class IDataStorage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def words(self):
        pass


class IStopWordFilter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def is_stop_word(self, word):
        pass


class IWordFrequencyCounter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def increment_count(self, word):
        pass

    @abc.abstractmethod
    def sorted(self):
        pass


class DataStorageManager(IDataStorage):
    _data = ""

    def __init__(self, path_to_file):
        with open(path_to_file) as f:
            self._data = f.read()
        pattern = re.compile(r"[\W_]+")
        self._data = pattern.sub(" ", self._data).lower()
        self._data = "".join(self._data).split()

    def words(self):
        return self._data


class StopWordManager(IStopWordFilter):
    _stop_words = []

    def __init__(self):
        with open("stop_words.txt") as f:
            self._stop_words = f.read().split(",")
        self._stop_words.extend(list(string.ascii_lowercase))

    def is_stop_word(self, word):
        return word in self._stop_words


class WordFrequencyManager(IWordFrequencyCounter):
    _word_freqs = {}

    def increment_count(self, word):
        self._word_freqs[word] = (
            1 if word not in self._word_freqs else self._word_freqs[word] + 1
        )

    def sorted(self):
        return sorted(
            self._word_freqs.items(), key=operator.itemgetter(1), reverse=True
        )


# using register or in class definition
# IDataStorage.register(DataStorageManager)
# IStopWordFilter.register(StopWordManager)
# IWordFrequencyCounter.register(WordFrequencyManager)


class WordFrequencyController:
    def __init__(self, path_to_file):
        self._storage = DataStorageManager(path_to_file)
        self._stop_word_manager = StopWordManager()
        self._word_freq_counter = WordFrequencyManager()

    def run(self):
        for w in self._storage.words():
            if not self._stop_word_manager.is_stop_word(w):
                self._word_freq_counter.increment_count(w)
        word_freqs = self._word_freq_counter.sorted()
        for w, c in word_freqs[0:25]:
            print(w, " - ", c)


@iter_profile(1000)
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
    WordFrequencyController(args["file"]).run()


if __name__ == "__main__":
    main()
