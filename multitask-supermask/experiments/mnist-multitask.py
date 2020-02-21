from main import main as run
from args import args


def main():
    args.set = ['MNIST32', 'FashionMNIST32', 'CIFAR10', 'CIFAR100']
    args.multigpu = [1]
    args.model = 'Conv4'
    args.conv_type = 'MultitaskMaskConv'
    args.bn_type = 'NonAffineBN'
    args.conv_init = 'kaiming_uniform'
    args.name = 'id=mnist-multitask'
    args.hamming = True
    run()

if __name__ == '__main__':
    main()