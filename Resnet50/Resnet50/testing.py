import residual_block
import resnet50
import training
import validation

# test 
model.eval()
test_loss = 0
test_correct = 0.0
test_total = 0.0
label_true = []
label_preds = []
with torch.no_grad():
    for images, labels in testloader:
        images = images.to('cuda')
        labels = labels.to('cuda')  

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # For multilabel, modify threshold
        test_correct += (predictions == labels).sum().item() 
        test_total += labels.numel() 
        label_true.append(labels)       #when we use it on cpu for inference, this might give an error 
        label_preds.append(predictions) #if gpu not available. So when the variable is used, need to do var_name.to('cpu")

    print(f"Epoch [{epoch+1}/{num_epochs}], test Loss: {test_loss/len(testloader)}")
test_accuracy = test_correct / test_total * 100
print(f" test Accuracy: {test_accuracy}")