from oneflow.utils.data import DataLoader
from flowvision import transforms
from flowvision import datasets
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.CIFAR10(
    root="/data",
    train=True,
    download=True,
    transform=transform_train
)
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True,
                                num_workers=1)#, persistent_workers=(parallel_workers > 0))
for batch, (X, y) in enumerate(train_dataloader):
    print(X)


