import residual_block
import resnet50
import testing
import training
import validation
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



#defined in other files
#traininging_loss
#validation_loss
#test_loss  
#training_accuracy 
#validation_accuracy
#test_accuracy   
#label_true
#label_preds



# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='training Loss', color='blue')
plt.plot(validation_loss, label='validation Loss', color='green')
plt.plot(test_loss, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(training_accuracy, label='Training Accuracy', color='blue')
plt.plot(validation_accuracy, label='Validation Accuracy', color='green')
plt.plot(test_accuracy, label='Test Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# F-1, recall, precision...
report = classification_report(label_true, label_preds, zero_division=0) 

