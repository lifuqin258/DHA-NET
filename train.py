import torch.optim as optim
from Fusion_network import *
from data_loader import *
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

train_acc_list = []
test_acc_list = []

start_time = time.time()

model = DHA_NET(channels=256)
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000005)

# 训练模型
for epoch in range(60):
    running_loss = 0.0
    correct = 0
    total = 0
    print('Epoch {}/{}'.format(epoch, 60))
    model.train()
    for data in DHA_NET_train_dataloader:
        vision_inputs, touch_inputs, labels = data
        if torch.cuda.is_available():
            vision_inputs = vision_inputs.cuda()
            touch_inputs = touch_inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(vision_inputs, touch_inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_acc = 100 * correct / total
    print(f'Train Accuracy of the model on the train images: {train_acc} %')
    train_acc_list.append(train_acc)
    torch.save(model.state_dict(), 'G:\\model\\aunoglcm.pth')


    correct = 0
    total = 0
    model.eval()
    all_preds = []
    all_labels = []
    correctly_classified = []
    true_labels_correct = []
    with torch.no_grad():
        for data in DHA_NET_test_dataloader:
            vision_inputs, touch_inputs, labels = data
            if torch.cuda.is_available():
                vision_inputs = vision_inputs.cuda()
                touch_inputs = touch_inputs.cuda()
                labels = labels.cuda()
            outputs = model(vision_inputs, touch_inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                if predicted[i] == labels[i]:
                    correctly_classified.append(predicted[i].item())
                    true_labels_correct.append(labels[i].item())

    test_acc = 100 * correct / total
    print(f'Test Accuracy of the model on the test images: {test_acc} %')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    test_acc_list.append(test_acc)
    test_precision = precision_score(all_labels, all_preds, average='weighted')
    test_recall = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1 Score: {test_f1}')
print('Finished Training')


end_time = time.time()
elapsed_time = end_time - start_time
print('Total training time: {:.2f} seconds'.format(elapsed_time))

plt.figure(figsize=(12, 6))
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch\nTotal training time: {:.2f} seconds'.format(elapsed_time))
plt.legend()
plt.grid(True)
plt.show()