import csv
import viz.utils as utils

import matplotlib.pyplot as plt


def read_csv_files(files, keys):
    d = {}
    for f in files:
        with open(f, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                name = row["Name"]
                d[name] = {}
                for k in keys:
                    d[name][k] = utils.try_float(row[k])
    return d


def main():
    layers = [
        "module.convs.0",
        "module.convs.2",
        "module.convs.5",
        "module.convs.7",
        "module.linear.0",
        "module.linear.2",
        "module.linear.4",
    ]
    rawdata = read_csv_files(
        [
            #"/home/mitchnw/git/connectome/pytorch/supermasks/experiments/hamming_distances/hamming.csv"
            "/Users/mitchnw/git/connectome/pytorch/supermasks/experiments/hamming_distances/hamming.csv"
        ],
        ["Distance"],
    )

    # postprocess the data
    data = {}
    for k, v in rawdata.items():
        d = utils.id_to_dict(k)
        d['layer'] = layers.index(d['layer'])
        tasks = [d['t1'].split('-')[0], d['t2'].split('-')[0]]
        tasks.sort()
        d['tasks'] = '-'.join(tasks)
        #del d['t1']
        #del d['t2']
        del d['rep']
        del d['try']
        data[utils.dict_to_id(d)] = v

    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=16)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(10, 10))
    ax1 = axlist
    utils.add_curves(
        ax1,
        data,
        x="layer",
        y="Distance",
        key="tasks",
        vals=['INIT-MNIST32', 'FashionMNIST32-INIT','CIFAR10-INIT', 'CIFAR100-INIT'],
        outlier=lambda x1,x2,x3: False,
    )
    ax1.set_ylabel('% Difference')
    ax1.set_xlabel('Layer')
    ax1.set_title('INIT Comparison')
    fig.subplots_adjust(top=0.9, left=0.14, right=0.97,
                        bottom=0.24)  # create some space below the plots by increasing the bottom-value
    ax1.legend(loc='upper center', bbox_to_anchor=(0.44, -0.12), ncol=2)

    #fig.show()
    plt.savefig('INIT_Comparison')

    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(10, 10))
    ax1 = axlist
    utils.add_curves(
        ax1,
        data,
        x="layer",
        y="Distance",
        key="tasks",
        vals=['MNIST32-MNIST32', 'FashionMNIST32-FashionMNIST32','CIFAR10-CIFAR10', 'CIFAR100-CIFAR100'],
        outlier=lambda x1,x2,x3: False,
    )
    ax1.set_ylabel('% Difference')
    ax1.set_xlabel('Layer')
    ax1.set_title('SELF Comparison')
    fig.subplots_adjust(top=0.9, left=0.14, right=0.97,
                        bottom=0.24)  # create some space below the plots by increasing the bottom-value
    ax1.legend(loc='upper center', bbox_to_anchor=(0.44, -0.12), ncol=2)

    #fig.show()
    plt.savefig('SELF_Comparison')

    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(10, 10))
    ax1 = axlist
    utils.add_curves(
        ax1,
        data,
        x="layer",
        y="Distance",
        key="tasks",
        vals=['MNIST32-MNIST32', 'FashionMNIST32-MNIST32', 'CIFAR10-MNIST32', 'CIFAR100-MNIST32', 'INIT-MNIST32'],
        outlier=lambda x1,x2,x3: False,
    )
    ax1.set_ylabel('% Difference')
    ax1.set_xlabel('Layer')
    ax1.set_title('MNIST32 Comparison')
    fig.subplots_adjust(top=0.9, left=0.14, right=0.97,
                        bottom=0.24)  # create some space below the plots by increasing the bottom-value
    ax1.legend(loc='upper center', bbox_to_anchor=(0.44, -0.12), ncol=2)

    #fig.show()
    plt.savefig('MNIST32_Comparison')

    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(11, 11))
    ax1 = axlist
    utils.add_curves(
        ax1,
        data,
        x="layer",
        y="Distance",
        key="tasks",
        vals=['FashionMNIST32-MNIST32', 'FashionMNIST32-FashionMNIST32', 'CIFAR10-FashionMNIST32', 'CIFAR100-FashionMNIST32', 'FashionMNIST32-INIT'],
        outlier=lambda x1,x2,x3: False,
    )
    ax1.set_ylabel('% Difference')
    ax1.set_xlabel('Layer')
    ax1.set_title('FashionMNIST32 Comparison')
    fig.subplots_adjust(top=0.9, left=0.14, right=0.97,
                        bottom=0.24)  # create some space below the plots by increasing the bottom-value
    ax1.legend(loc='upper center', bbox_to_anchor=(0.44, -0.12), ncol=2)

    #fig.show()
    plt.savefig('FashionMNIST32_Comparison')

    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(10, 10))
    ax1 = axlist
    utils.add_curves(
        ax1,
        data,
        x="layer",
        y="Distance",
        key="tasks",
        vals=['CIFAR10-MNIST32', 'CIFAR10-FashionMNIST32', 'CIFAR10-CIFAR10', 'CIFAR10-CIFAR100', 'CIFAR10-INIT'],
        outlier=lambda x1,x2,x3: False,
    )
    ax1.set_ylabel('% Difference')
    ax1.set_xlabel('Layer')
    ax1.set_title('CIFAR10 Comparison')
    fig.subplots_adjust(top=0.9, left=0.14, right=0.97,
                        bottom=0.24)  # create some space below the plots by increasing the bottom-value
    ax1.legend(loc='upper center', bbox_to_anchor=(0.44, -0.12), ncol=2)

    #fig.show()
    plt.savefig('CIFAR10_Comparison')

    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(10, 10))
    ax1 = axlist
    utils.add_curves(
        ax1,
        data,
        x="layer",
        y="Distance",
        key="tasks",
        vals=['CIFAR100-MNIST32', 'CIFAR100-FashionMNIST32', 'CIFAR10-CIFAR100', 'CIFAR100-CIFAR100', 'CIFAR100-INIT'],
        outlier=lambda x1,x2,x3: False,
    )
    ax1.set_ylabel('% Difference')
    ax1.set_xlabel('Layer')
    ax1.set_title('CIFAR100 Comparison')
    fig.subplots_adjust(top=0.9, left=0.14, right=0.97,
                        bottom=0.24)  # create some space below the plots by increasing the bottom-value
    ax1.legend(loc='upper center', bbox_to_anchor=(0.44, -0.12), ncol=2)

    #fig.show()
    plt.savefig('CIFAR100_Comparison')


if __name__ == "__main__":
    main()
