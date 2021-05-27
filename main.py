import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-x", "--xxxx", help="Exemple argument", type=int, nargs=2)
    args = parser.parse_args()
