import os, sys


def main():
    path = [
        "./transformer",
        ]

    for p in path :
        if p not in sys.path:
            sys.path.append(path)
            # print(p)


main()


