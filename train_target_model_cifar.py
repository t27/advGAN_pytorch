import torch
from torch.utils.data import dataloader
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models import CIFAR_Target_Net


if __name__ == "__main__":
    use_cuda = True
    image_nc = 1
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=6
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=6
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # training the target model
    target_model = CIFAR_Target_Net().to(device)
    target_model.train()

    # opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001)
    opt_model = torch.optim.SGD(target_model.parameters(), lr=0.001, momentum=0.9)
    epochs = 100
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            # opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001)
            opt_model = torch.optim.SGD(
                target_model.parameters(), lr=0.0001, momentum=0.9
            )
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels)
            loss_epoch += loss_model
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()

        print(
            "loss in epoch %d: %f" % (epoch, loss_epoch.item() / len(train_dataloader))
        )

        if epoch % 10 == 0:
            target_model.eval()

            num_correct = 0
            for i, data in enumerate(test_dataloader, 0):
                test_img, test_label = data
                test_img, test_label = test_img.to(device), test_label.to(device)
                pred_lab = torch.argmax(target_model(test_img), 1)
                num_correct += torch.sum(pred_lab == test_label, 0)
            target_model.train()
            print("accuracy in testing set: %f\n" % (num_correct.item() / len(testset)))

            # save model
            targeted_model_file_name = f"./CIFAR_target_model_epoch_{epoch}.pth"
            torch.save(target_model.state_dict(), targeted_model_file_name)
            target_model.eval()

    targeted_model_file_name = f"./CIFAR_target_model.pth"
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab == test_label, 0)

    print("accuracy in testing set: %f\n" % (num_correct.item() / len(testset)))
