# author: choi sugil
# date: 2023.10.13 version: 1.0.0 license: MIT brief: keyward
# description:
import argparse
import dotenv
import os
import re
import operator
import string

# fix current working directory '/python' folder
# dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv.find_dotenv())
CWD = os.environ["CWD_python"]
print(CWD)
os.chdir(CWD)


class WordFrequencyFramework:
    _load_event_handlers = []
    _dowork_event_handlers = []
    _end_event_handlers = []

    def register_for_load_event(self, handler):
        self._load_event_handlers.append(handler)

    def register_for_dowork_event(self, handler):
        self._dowork_event_handlers.append(handler)

    def register_for_end_event(self, handler):
        self._end_event_handlers.append(handler)

    def run(self, path_to_file):
        for h in self._load_event_handlers:
            h(path_to_file)
        for h in self._dowork_event_handlers:
            h()
        for h in self._end_event_handlers:
            h()


class StopWordFilter:
    _stop_words = []

    def __init__(self, wfapp):
        wfapp.register_for_load_event(self.__load)

    def __load(self, ignore):
        with open("stop_words.txt") as f:
            self._stop_words = f.read().split(",")
        self._stop_words.extend(list(string.ascii_lowercase))

    def is_stop_word(self, word):
        return word in self._stop_words


class DataStrorage:
    _data = ""
    _stop_word_filter = None
    _word_event_handlers = []

    def __init__(self, wfapp, stop_word_filter: StopWordFilter):
        self._stop_word_filter = stop_word_filter
        wfapp.register_for_load_event(self.__load)
        wfapp.register_for_dowork_event(self.__produce_words)

    def __load(self, path_to_file):
        with open(path_to_file) as f:
            self._data = f.read()
        pattern = re.compile(r"[\W_]+")
        self._data = pattern.sub(" ", self._data).lower()
        self._data = "".join(self._data).split()

    def __produce_words(self):
        for w in self._data:
            if not self._stop_word_filter.is_stop_word(w):  # type: ignore
                for h in self._word_event_handlers:
                    h(w)

    def register_for_word_event(self, handler):
        self._word_event_handlers.append(handler)


class WordFrequencyCounter:
    _word_freqs = {}

    def __init__(self, wfapp, data_storage):
        data_storage.register_for_word_event(self.__increment_count)
        wfapp.register_for_end_event(self.__print_freqs)

    def __increment_count(self, word):
        self._word_freqs[word] = (
            1 if word not in self._word_freqs else self._word_freqs[word] + 1
        )

    def __print_freqs(self):
        word_freqs = sorted(
            self._word_freqs.items(), key=operator.itemgetter(1), reverse=True
        )
        for w, c in word_freqs[0:25]:
            print(w, " - ", c)


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
    wfapp = WordFrequencyFramework()
    stop_word_filter = StopWordFilter(wfapp)
    data_storage = DataStrorage(wfapp, stop_word_filter)
    WordFrequencyCounter(wfapp, data_storage)
    wfapp.run(args["file"])


if __name__ == "__main__":
    main()
