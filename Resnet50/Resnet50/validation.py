import residual_block
import resnet50
import training

# Validation 
model.eval()
validation_loss = 0
validation_correct = 0.0
validation_total = 0.0
with torch.no_grad():
    for images, labels in validation_loader:
        images = images.to('cuda')
        labels = labels.to('cuda')  

        outputs = model(images)
        loss = criterion(outputs, labels)
        validation_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # For multilabel, modify threshold
        validation_correct += (predictions == labels).sum().item() 
        validation_total += labels.numel() 

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {validation_loss/len(validation_loader)}")
validation_accuracy = validation_correct / validation_total * 100
print(f" Validation Accuracy: {validation_accuracy}")