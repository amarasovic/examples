from __future__ import print_function
import os
import pathlib
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import data
import utils
from args import args


def train(model, writer, train_loader, optimizer, criterion, epoch, task):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

            t = (len(train_loader) * epoch + batch_idx) * args.batch_size
            writer.add_scalar("train_{}/loss".format(task), loss.item(), t)


def test(model, writer, criterion, test_loader, epoch, task):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = float(correct) / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: ({:.4f}%)\n".format(
            test_loss, test_acc
        )
    )

    writer.add_scalar("test_{}/loss".format(task), test_loss, epoch)
    writer.add_scalar("test_{}/acc".format(task), test_acc, epoch)

    return test_acc


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/{args.name}~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + "~try={}".format(str(i))
            break
        i += 1

    (run_base_dir / "settings.txt").write_text(str(args))

    # get the datasets.
    count = {}
    for i in range(len(args.set)):
        count[args.set[i]] = 1
        for j in range(i+1, len(args.set)):
            if args.set[j] == args.set[i]:
                args.set[j] = args.set[j] + '-v{}'.format(count[args.set[i]] + 1)
                count[args.set[i]] += 1

    sets = {set: getattr(data, set.split('-')[0])() for set in args.set}

    best_acc1 = {set: 0.0 for set in args.set}
    curr_acc1 = {set: 0.0 for set in args.set}

    model = utils.get_model()
    model = utils.set_gpu(model)

    optimizers = {}
    schedulers = {}
    for set in args.set:
        params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.split(".")[-1] in args.set and n.split(".")[-1] != set:
                continue
            params.append(p)

        optimizers[set] = optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
        schedulers[set] = CosineAnnealingLR(optimizers[set], T_max=args.epochs)


    criterion = nn.CrossEntropyLoss().to(args.device)


    writer = SummaryWriter(log_dir=run_base_dir)

    # Save the initial state
    torch.save(
        {
            "epoch": 0,
            "model": args.model,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "curr_acc1": curr_acc1,
            "args": args,
        },
        run_base_dir / "init.pt",
    )

    if args.hamming:
        utils.write_hamming(writer, model, 0)

    for epoch in range(1, args.epochs + 1):
        for task, loader in sets.items():
            model.apply(lambda m: setattr(m, "task", task))
            train(model, writer, loader.train_loader, optimizers[task], criterion, epoch, task)
            curr_acc1[task] = test(
                model, writer, criterion, loader.val_loader, epoch, task
            )
            if curr_acc1[task] > best_acc1[task]:
                best_acc1[task] = curr_acc1[task]
            schedulers[task].step()

        if epoch == args.epochs or (
            args.save_every > 0 and (epoch % args.save_every) == 0
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "arch": args.model,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "curr_acc1": curr_acc1,
                    "args": args,
                },
                run_base_dir / "epoch_{}.pt".format(epoch),
            )

        if args.hamming:
            utils.write_hamming(writer, model, epoch)

    for set in args.set:
        utils.write_result_to_csv(
            name=args.name + "~task={}".format(set),
            curr_acc1=curr_acc1[task],
            best_acc1=best_acc1[task],
        )

    if args.hamming:
        utils.log_hamming(model)


if __name__ == "__main__":
    main()
