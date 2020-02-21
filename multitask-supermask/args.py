import argparse

args = None


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0001,
        metavar="M",
        help="Weight decay (default: 0.0001)",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--workers", type=int, default=4, help="how many cpu workers")
    parser.add_argument(
        "--output-size", type=int, default=100, help="how many cpu workers"
    )

    parser.add_argument(
        "--save-every", type=int, default=-1, help="How often to save the model."
    )
    parser.add_argument("--name", type=str, default="default", help="Experiment id.")
    parser.add_argument(
        "--data", type=str, default="/home/mitchnw/data", help="Location to store data"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/home/mitchnw/ssd/checkpoints/connectome",
        help="Location to store data",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="how sparse is each layer"
    )
    parser.add_argument(
        "--width-mult", type=float, default=1.0, help="how wide is each layer"
    )
    parser.add_argument(
        "--conv_type", type=str, default="StandardConv", help="Type of conv layer"
    )
    parser.add_argument(
        "--bn_type", type=str, default="StandardBN", help="Type of batch norm layer."
    )
    parser.add_argument(
        "--conv_init",
        type=str,
        default="default",
        help="How to initialize the conv weights.",
    )
    parser.add_argument("--model", type=str, help="Type of model.")
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )
    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--hamming", action="store_true", default=False)

    parser.add_argument("--set", nargs="+", default=[])

    args = parser.parse_args()

    return args


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
