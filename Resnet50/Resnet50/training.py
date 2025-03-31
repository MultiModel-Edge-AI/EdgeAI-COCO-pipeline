import torch
from torch import nn
from residual_block 
from resnet50 
#from data_file import train and test dataloader

# Create an object of class ResNet50
model = ResNet50(num_classes=80)  #  COCO dataset with 80 classes
# print(model)    #to be decided, if we want to show the architecture
num_epochs = 100   #to be modified

# Optimization, to be added to Milestone 2 model optimization part. 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) Option 2
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, momentum=0.9, weight_decay=1e-4) Option 3

criterion = nn.BCEWithLogitsLoss()


for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    correct = 0
    total = 0

    for images, labels in trainloader: 
        images = images.to('cuda')   #use GPU
        labels = labels.to('cuda')


        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # For multilabel, modify threshold
        correct += (predictions == labels).sum().item() 
        total += labels.numel()  
        
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss/len(trainloader)}")
    training_accuracy = correct / total * 100
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {training_accuracy}")


