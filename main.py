from models import mynet
from data_loader import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

train_transform = transforms.Compose([
    # transforms.Resize(112),
    transforms.RandomCrop(90),
    transforms.Resize(112),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465, 0.5), (0.2023, 0.1994, 0.2010, 0.2)),
])

val_transform = transforms.Compose([
    transforms.CenterCrop(90),
    transforms.Resize(112),

    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465, 0.5), (0.2023, 0.1994, 0.2010, 0.2)),
])

if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Loading data...")

    # weights = [2 if label == 1 else 1 for data, label in train_dataset]
    # sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=False)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=sampler)

    train_dataset = CASIASURFDataset(split="train", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_dataset = CASIASURFDataset(split="val", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    # net = mynet.MyNetv2CIFAR(os.path.join(MAIN_DIR, "pretrained", "MyNet_CIFAR10.pth")).to(device)
    net = mynet.MyNetv2CIFAR(os.path.join(MAIN_DIR, "pretrained", "MyNet_CIFAR10_112.pth"))
    # net.load_state_dict(torch.load(os.path.join(MAIN_DIR, "pretrained", "MyNet_CIFAR10_112.pth")))
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

    print("Training started.")
    best_acc = 0.0
    best_net_state_dict = None
    for epoch in range(50):  # loop over the dataset multiple times

        train_loss, train_acc = 0.0, 0.0
        train_correct, train_total = 0, 0

        for i, data in enumerate(train_loader, 0):
            net = net.train()
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # record statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            if (i != 0 and i % 100 == 0) or i + 1 == len(train_loader):

                val_loss, val_acc = 0.0, 0.0
                val_correct, val_total = 0, 0
                net = net.eval()
                with torch.no_grad():
                    for _, data in enumerate(val_loader, 0):
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)

                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                    val_acc = val_correct / val_total

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_net_state_dict = net.state_dict()
                # print(train_total, val_total)
                print(f"Epoch {epoch:3d} Batch {i:3d}: training loss = {train_loss / (i + 1):.4f}; training accuracy = {train_correct / train_total:.4f}; validation loss = {val_loss / len(val_loader) :.4f}; validation accuracy = {val_correct / val_total:.4f}")

        # train_acc = train_correct / train_total


        # val_correct, val_total = 0, 0
        # net = net.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(val_loader, 0):
        #         inputs, labels = data
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #
        #         outputs = net(inputs)
        #         loss = criterion(outputs, labels)
        #         val_loss += loss.item()
        #         _, predicted = outputs.max(1)
        #
        #         val_total += labels.size(0)
        #         val_correct += predicted.eq(labels).sum().item()



        #print(f"Epoch {epoch}: training loss = {train_loss / train_total:.4f}; training accuracy = {train_acc:.4f}; validation loss = {val_loss / val_total:.4f}; validation accuracy = {val_acc:.4f}")

    print(f"Training finished. Best accuracy: {best_acc}")
    # torch.save(best_net_state_dict, os.path.join(MAIN_DIR, "pretrained", "MyNet_CIFAR10_112.pth"))
