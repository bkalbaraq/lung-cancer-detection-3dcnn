import argparse
from . import preprocess_3d, train_3dcnn, eval_metrics, infer_scan

def main():
    parser = argparse.ArgumentParser(description="LIDC 3D pipeline")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("prep")
    sub.add_parser("train")
    sub.add_parser("eval")
    sub.add_parser("infer")

    args = parser.parse_args()
    if args.cmd == "prep":
        preprocess_3d.main()
    elif args.cmd == "train":
        train_3dcnn.main()
    elif args.cmd == "eval":
        eval_metrics.main()
    elif args.cmd == "infer":
        infer_scan.main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
