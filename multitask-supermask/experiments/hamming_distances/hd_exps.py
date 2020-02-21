from main import main as run
from args import args

def main():

    for seed in [1996, 1997, 1998]:
        args.set = ['MNIST32', 'FashionMNIST32', 'CIFAR10', 'CIFAR100']
        args.seed = seed
        args.multigpu = [1]
        args.model = 'Conv4'
        args.conv_type = 'MultitaskMaskConv'
        args.bn_type = 'NonAffineBN'
        args.conv_init = 'kaiming_uniform'
        args.name = 'id=hd~rep=0~seed={}'.format(seed)
        args.hamming = True
        args.log_dir = '/home/mitchnw/ssd/checkpoints/connectome/hd'
        run()

if __name__ == '__main__':
    main()