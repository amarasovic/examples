import csv
import matplotlib.pyplot as plt
import numpy as np


def try_float(x):
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x


# keys is a list of strings k_1,...,k_n
# returns a dict d where d[name][k_i] = val corresponding to k_i
def read_csv_files(files, keys):
    d = {}
    for f in files:
        with open(f, mode="r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                name = row["Name"]
                d[name] = {}
                for k in keys:
                    d[name][k] = try_float(row[k])
    return d


def id_to_dict(id):
    l = [x.split("=") for x in id.split("~")]
    return {x[0].strip(): try_float(x[1]) for x in l}

def dict_to_id(dict):
    return '~'.join('{}={}'.format(k,v) for k,v in dict.items())


def query(data, x, y, outlier=lambda x, y, d: False, **kwargs):
    out = {}
    for id in data:
        dict = id_to_dict(id)
        valid = True
        for k, v in kwargs.items():
            if k not in dict or dict[k] != v:
                valid = False
                break
        if valid:
            if x in dict:
                x_ = dict[x]
                y_ = data[id][y]
                if not outlier(x_, y_, data):
                    if x_ not in out:
                        out[x_] = []
                    out[x_].append(y_)
                else:
                    print(y_)

    return out


def add_curve(ax, data, label, ls="-", marker=None, color=None):
    xs = np.array(sorted([k for k in data]))
    ys = np.array([np.mean(data[x]) for x in xs])
    std = np.array([np.std(data[x]) for x in xs])
    ax.plot(xs, ys, label=label, linestyle=ls, marker=marker, color=color)
    ax.fill_between(xs, ys - std, ys + std, alpha=0.5, color=color)


def add_curves(ax, data, x, y, key, vals, outlier=lambda x, y, d: False, colors=None, ls=None, prefix="", **kwargs):

    if colors is None:
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    if ls is None:
        ls = ["-"]
    i = 0
    for val in vals:
        kwargs[key] = val
        q = query(data, x=x, y=y, outlier=outlier, **kwargs)
        add_curve(ax, q, "{}{}={}".format(prefix, key, val), ls=ls[i % len(ls)], color=colors[i % len(colors)])
        i += 1


def main():
    data = read_csv_files(
        ["/home/mitchnw/git/exp2/results/mnist_dd_dec18.csv"], ["Current Val Top 1"]
    )
    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(8, 8))
    ax1 = axlist
    add_curves(
        ax1,
        data,
        x="wm",
        y="Current Val Top 1",
        key="ln",
        vals=[0.025, 0.05, 0.075],
        outlier=lambda v: v < 50,
        id="mnist-dd",
    )
    fig.show()


def eg1():
    # print(id_to_dict("id=mnist-lottery~wm=1.6~ew=0.8~seed=1999"))
    data = read_csv_files(
        ["/home/mitchnw/git/exp2/results/mnist_dd_dec18.csv"], ["Current Val Top 1"]
    )
    out = query(
        data,
        x="wm",
        y="Current Val Top 1",
        outlier=lambda v: v < 50,
        ln=0.1,
        id="mnist-dd",
    )
    out1 = query(
        data,
        x="wm",
        y="Current Val Top 1",
        outlier=lambda v: v < 50,
        ln=0.075,
        id="mnist-dd",
    )
    out2 = query(
        data,
        x="wm",
        y="Current Val Top 1",
        outlier=lambda v: v < 50,
        ln=0.05,
        id="mnist-dd",
    )

    fig, axlist = plt.subplots(1, 1, sharey=False, figsize=(8, 8))
    ax1 = axlist

    # add_curve(ax1, out, "ln=0.1")
    add_curve(ax1, out1, "ln=0.075")
    add_curve(ax1, out2, "ln=0.05")

    ax1.set_ylim([82.5, 99])
    fig.show()


if __name__ == "__main__":
    main()
