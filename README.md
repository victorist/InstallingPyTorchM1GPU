# Installing and runing PyTorch on M1 GPUs (Apple metal/MPS)
On May 18, 2022, PyTorch and Apple teams, having done a great job, made it possible for the PyTorch framework to work on M1 graphics cores.

# METAL ACCELERATION

Accelerated GPU training is enabled using Apple’s Metal Performance Shaders (MPS) as a backend for PyTorch. The MPS backend extends the PyTorch framework, providing scripts and capabilities to set up and run operations on Mac. MPS optimizes compute performance with kernels that are fine-tuned for the unique characteristics of each Metal GPU family. The new device maps machine learning computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS.

# TRAINING BENEFITS ON APPLE SILICON

Every Apple silicon Mac has a unified memory architecture, providing the GPU with direct access to the full memory store. This makes Mac a great platform for machine learning, enabling users to train larger networks or batch sizes locally. This reduces costs associated with cloud-based development or the need for additional local GPUs. The Unified Memory architecture also reduces data retrieval latency, improving end-to-end performance.

In the graphs below, you can see the performance speedup from accelerated GPU training and evaluation compared to the CPU baseline:

# On installing PyTorch for M1

Anaconda is the recommended package manager as it installs all dependencies.

As of May 28, 2022, the PyTorch Stable version (1.11.0) only works with x86 and M1. For Apple GPU, you need to use the PyTorch Preview (Nightly) version.

**Important!** MPS acceleration is available in **macOS 12.3+**

At the time of writing this instruction (May 30, 2022), the recommended command line for conda was reporting errors (torchaudio was not found, there are many package inconsistencies for torchvision).


__Original comand line. The command fails due to errors!__

    conda install pytorch torchvision torchaudio -c pytorch-nightly

I was able to install PyTorc for MPS using the following sequence of commands:

    conda install pytorch -c pytorch-nightly
    pip3 install --pre torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

**Verify your installation**

    Python 3.10.4 (main, Mar 31 2022, 03:37:37) [Clang 12.0.0 ] on darwinType "help", "copyright", "credits" or "license" for more information.
    >>> import torch; torch.backends.mps.is_available()
    True
    >>>

**Sample program to test PyTorch on MPS**

Sample program to test PyTorch on MPS
Sample program - in file `main.py`

    from __future__ import print_function
    import argparse
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.optim.lr_scheduler import StepLR


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output


    def train(args, model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break


    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    def main():
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')
        parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--dry-run', action='store_true', default=False,
                            help='quickly check a single pass')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
        args = parser.parse_args()
        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        # device = torch.device("cuda" if use_cuda else "cpu")
        print("MPS availablity:",torch.backends.mps.is_available())
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ## and then move your model and data to the device before you train or eval. Have fun folks!

        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


    if __name__ == '__main__':
        main()


During the execution of this program, you can observe the loading of M1 graphics cores.

![M1 GPU process](m1-gpu-process.png)