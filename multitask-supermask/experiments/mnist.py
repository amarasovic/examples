from main import main as run
from args import args

def main():
    args.set = ['CIFAR10']
    args.multigpu = [1]
    args.model = 'Conv4'
    args.conv_type = 'MaskConv'
    args.bn_type = 'NonAffineBN'
    args.conv_init = 'kaiming_uniform'
    args.name = 'id=mnist'
    args.epochs = 100

    run()

if __name__ == '__main__':
    main()