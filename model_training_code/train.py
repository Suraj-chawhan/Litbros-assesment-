#import
import torch
from tqdm import tqdm
import time
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import zipfile
import os



""" Extract folder mask dataset

zip_path = "/content/data.zip"
extract_path = "/content/data"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
os.listdir(extract_path)



"""


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

dataset = ImageFolder(root=f"{extract_path}/data", transform=transform)
print("Classes:", dataset.classes)  # should now show ['mask', 'not_mask']

# Split train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)




#Model Creation



class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()

        # Convolutional Block 1
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Convolutional Block 2
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.fc(x)
        return x




#model reference and loss function and optimizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=MaskCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def binary_accuracy(preds, labels):
    preds = torch.sigmoid(preds)
    preds = torch.round(preds)
    correct = (preds == labels).float()
    return correct.sum() / len(correct)




EPOCHS =15
start_time = time.time()

for epoch in range(EPOCHS):

    # Training

    model.train()
    train_loss, train_acc = 0, 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = binary_accuracy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc.item()

        loop.set_postfix(loss=loss.item(), acc=acc.item())


    # Test data loss accuracy
    model.eval()
    test_loss, test_acc = 0, 0
    loop_test = tqdm(test_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Validation", leave=False)
    with torch.no_grad():
        for images, labels in loop_test:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = binary_accuracy(outputs, labels)

            test_loss += loss.item()
            test_acc += acc.item()

            loop_test.set_postfix(test_loss=loss.item(), test_acc=acc.item())


    # Epoch Results

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc/len(train_loader):.4f} || "
          f"Test Loss: {test_loss/len(test_loader):.4f} | "
          f"Test Acc: {test_acc/len(test_loader):.4f}")

end_time = time.time()
print(f"\n Training completed in {(end_time - start_time)/60:.2f} minutes for {EPOCHS} epochs")





#model save



def save_model(model, path="mask_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")


save_model(model)
