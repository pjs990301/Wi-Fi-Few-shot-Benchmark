import torch

def train(model, tensor_loader, num_epochs, learning_rate, criterion, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    accuracy_history = []  # Accuracy 기록을 위한 리스트
    loss_history = []  # Loss 기록을 위한 리스트

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        # print(len(tensor_loader))
        for data in tensor_loader:

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            
            # print(inputs.shape)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)

        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        accuracy_history.append(epoch_accuracy)
        loss_history.append(epoch_loss)
        optimizer.step()
        print('Epoch:{}, Accuracy:{:.5f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))

    return accuracy_history, loss_history


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    accuracy_history = []  # Accuracy 기록을 위한 리스트
    loss_history = []  # Loss 기록을 위한 리스트

    for data in tensor_loader:
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)

        # print(inputs)
        # print(labels)
        # print(inputs.shape)
        
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)

        accuracy_history.append(accuracy)
        loss_history.append(loss.item() * inputs.size(0))

        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)

        print('test_Accuracy:{:.5f}, test_Loss:{:.9f}'.format( float(accuracy), float(loss)))


    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print("validation accuracy:{:.5f}, loss:{:.9f}".format(float(test_acc), float(test_loss)))
    return test_acc, test_loss, accuracy_history, loss_history
