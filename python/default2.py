# author: choi sugil
# date: 2023.10.13 version: 1.0.0 license: MIT brief: keyward
# description: default main program with option parser and cwd, dotenv
import argparse
import dotenv
import os

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
    print(args)


if __name__ == "__main__":
    main()
