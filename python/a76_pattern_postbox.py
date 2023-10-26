# author: choi sugil
# date: 2023.10.13 version: 1.0.0 license: MIT brief: keyward
# description: postbox pattern
import argparse
import dotenv
import os
import operator
import string
import re
from a75_pattern_object import iter_profile

# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)


class DataStorageManager:
    _data = ""

    def dispatch(self, message):
        if message[0] == "init":
            return self._init(message[1])
        elif message[0] == "words":
            return self._words()
        else:
            raise Exception("Message not understood " + message[0])

    def _init(self, path_to_file):
        with open(path_to_file) as f:
            self._data = f.read()
        pattern = re.compile(r"[\W_]+")
        self._data = pattern.sub(" ", self._data).lower()

    def _words(self):
        return self._data.split()


class StopWordManager:
    _stop_words = []

    def dispatch(self, message):
        if message[0] == "init":
            return self._init()
        elif message[0] == "is_stop_word":
            return self._is_stop_word(message[1])
        else:
            raise Exception("Message not understood " + message[0])

    def _init(self):
        with open("stop_words.txt") as f:
            self._stop_words = f.read().split(",")
        self._stop_words.extend(list(string.ascii_lowercase))

    def _is_stop_word(self, word):
        return word in self._stop_words


class WordFrequencyManager:
    _word_freqs = {}

    def dispatch(self, message):
        if message[0] == "increment_count":
            return self._increment_count(message[1])
        elif message[0] == "sorted":
            return self._sorted()
        else:
            raise Exception("Message not understood " + message[0])

    def _increment_count(self, word):
        if word in self._word_freqs:
            self._word_freqs[word] += 1
        else:
            self._word_freqs[word] = 1

    def _sorted(self):
        return sorted(
            self._word_freqs.items(), key=operator.itemgetter(1), reverse=True
        )


class WordFrequencyController:
    def dispatch(self, message):
        if message[0] == "init":
            return self._init(message[1])
        elif message[0] == "run":
            return self._run()
        else:
            raise Exception("Message not understood " + message[0])

    def _init(self, path_to_file):
        self._storage_manager = DataStorageManager()
        self._stop_word_manager = StopWordManager()
        self._word_freq_manager = WordFrequencyManager()
        self._storage_manager.dispatch(["init", path_to_file])
        self._stop_word_manager.dispatch(["init"])

    def _run(self):
        for w in self._storage_manager.dispatch(["words"]):  # type: ignore
            if not self._stop_word_manager.dispatch(["is_stop_word", w]):
                self._word_freq_manager.dispatch(["increment_count", w])

        word_freqs = self._word_freq_manager.dispatch(["sorted"])
        for w, c in word_freqs[0:25]:  # type: ignore
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
    wfcontroller = WordFrequencyController()
    wfcontroller.dispatch(["init", args["file"]])
    wfcontroller.dispatch(["run"])


if __name__ == "__main__":
    main()
