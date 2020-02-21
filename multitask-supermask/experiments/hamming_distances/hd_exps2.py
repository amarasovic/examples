from main import main as run
from args import args

def main():
    datasets = ['MNIST32', 'FashionMNIST32', 'CIFAR10', 'CIFAR100']
    for dset in datasets:
        args.set = [dset, dset, dset]
        args.seed = 1996
        args.multigpu = [0]
        args.model = 'Conv4'
        args.conv_type = 'MultitaskMaskConv'
        args.bn_type = 'NonAffineBN'
        args.conv_init = 'kaiming_uniform'
        args.name = 'id=hd~rep={}'.format(dset)
        args.hamming = True
        args.log_dir = '/home/mitchnw/ssd/checkpoints/connectome/hd'
        run()

if __name__ == '__main__':
    main()