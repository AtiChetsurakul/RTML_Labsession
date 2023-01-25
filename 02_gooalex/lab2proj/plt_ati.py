import matplotlib.pyplot as plt
import torch


def plotaccloss(loss, acc, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(name)
    # ax1.plot(torch.Tensor(val_acc_log).cpu(),label = 'trainset')
    ax1.plot(torch.Tensor(loss).cpu(), label='val_set')
    ax1.legend()
    ax1.set_title('Loss vs Iteration')

    # ax2.plot(torch.Tensor(train_acc_log).cpu(),label = 'trainset')
    ax2.plot(torch.Tensor(acc).cpu(), label='val_set')
    ax2.legend()
    ax2.set_title('Accuracy vs Iteration')
    plt.show()
