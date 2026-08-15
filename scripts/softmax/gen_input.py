import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate row-major softmax input.")
    parser.add_argument("--rows", type=int, default=320)
    parser.add_argument("--cols", type=int, default=4096)
    args = parser.parse_args()

    print(args.rows, args.cols)
    for i in range(args.rows * args.cols):
        print(float(i % args.cols) / 100.0)


if __name__ == "__main__":
    main()
