import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net, CIFAR_Target_Net

use_cuda = True
image_nc = 3
batch_size = 128

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "./CIFAR_target_model_epoch_99.pth"
target_model = CIFAR_Target_Net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# load the generator of adversarial examples
pretrained_generator_path = "./models/netG_epoch_60.pth"
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_dataloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=6
)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print("CIFAR training dataset:")
print("num_correct: ", num_correct.item())
print(
    "accuracy of adv imgs in training set: %f\n" % (num_correct.item() / len(trainset))
)

# test adversarial examples in MNIST testing dataset
# mnist_dataset_test = torchvision.datasets.MNIST(
#     "./dataset", train=False, transform=transforms.ToTensor(), download=True
# )
# test_dataloader = DataLoader(
#     mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1
# )
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
test_dataloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=6
)
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print("num_correct: ", num_correct.item())
print("accuracy of adv imgs in testing set: %f\n" % (num_correct.item() / len(testset)))

