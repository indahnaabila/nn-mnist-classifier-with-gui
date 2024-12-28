import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

st.title("Neural Network MNIST Classifier")
st.set_option('deprecation.showPyplotGlobalUse', False)

def load_data():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./ANN/Data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./ANN/Data', train=False, download=True, transform=transform)
    return train_data, test_data

class MultilayerPerceptron(nn.Module):

    def __init__(self, in_sz, out_sz, layers):
        super().__init__()
        st.text(layers)
        self.fc_layers = nn.ModuleList()
        # Input layer to first hidden layer
        self.fc_layers.append(nn.Linear(in_sz, layers[0]))
        # Hidden layers
        for i in range(len(layers)-1):
            self.fc_layers.append(nn.Linear(layers[i], layers[i+1]))
        # Last hidden layer to output layer
        self.fc_layers.append(nn.Linear(layers[-1], out_sz))

    def forward(self, X):
        for layer in self.fc_layers[:-1]:
            X = F.relu(layer(X))
        X = self.fc_layers[-1](X)
        return F.log_softmax(X, dim=1)

# Define the Convolutional Network class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Function for model training and evaluation
def train(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for epoch in range(epochs):
        train_correct_count = 0
        test_correct_count = 0

        # Training
        model.train()
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X_train.view(-1, 28*28))
            loss = criterion(output, y_train)

            #y_pred = model(X_train.view(100, -1))
            #predicted = torch.max(y_pred.data, 1)[1]
            #batch_corr = (predicted == y_train).sum()
            #train_correct_count += batch_corr

            loss.backward()
            optimizer.step()

            #if batch_idx % 200 == 0:
                #st.write(f"Epoch: {epoch+1} Batch: {batch_idx} Loss: {loss.item()} Accuracy: {train_correct_count.item()*100/(100*batch_idx):7.3f}%")

            if batch_idx % 200 == 0:
                st.write(f"Epoch: {epoch+1} Batch: {batch_idx} Loss: {loss.item()}")


            _, predicted = torch.max(output, 1)
            train_correct_count += (predicted == y_train).sum().item()

        # Testing
        model.eval()
        with torch.no_grad():
            for batch_idx, (X_test, y_test) in enumerate(test_loader):
                output = model(X_test.view(-1, 28*28))
                _, predicted = torch.max(output.data, 1)
                test_correct_count += (predicted == y_test).sum().item()

        #train_loss = criterion(output, y_test)
        test_loss = criterion(output, y_test)
        train_accuracy = 100.0 * train_correct_count / len(train_loader.dataset)
        test_accuracy = 100.0 * test_correct_count / len(test_loader.dataset)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_correct.append(train_correct_count)
        test_correct.append(test_correct_count)

    return train_losses, test_losses, train_correct, test_correct

#function for cnn model training and evaluation 
def train_model(model, epochs, train_batch_size, train_loader, test_loader, criterion, optimizer):
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    for epoch in range(epochs):
        train_corr = 0
        test_corr = 0

        for batch, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            predicted = torch.max(output.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            train_corr += batch_corr
            loss.backward()
            optimizer.step()

            if batch % 600 == 0:
                st.write(f'Epoch: {epoch+1}, Batch: {batch+1}, Loss: {loss.item():.8f}, Accuracy: {train_corr.item()*100/(train_batch_size*(batch+1)):.3f}%')

        train_losses.append(loss.item())
        train_correct.append(train_corr.item())

        with torch.no_grad():
            for batch, (X_test, y_test) in enumerate(test_loader):
                output = model(X_test)
                predicted = torch.max(output.data, 1)[1]
                test_corr += (predicted == y_test).sum()

        loss = criterion(output, y_test)
        test_losses.append(loss.item())
        test_correct.append(test_corr.item())

    return train_losses, train_correct, test_losses, test_correct

# Evaluate the CNN model
def evaluate_model(model, test_loader, test_data):
    test_correct = 0
    for X_test, y_test in test_loader:
        output = model(X_test)
        predicted = torch.max(output.data, 1)[1]
        test_correct += (predicted == y_test).sum()
    accuracy = test_correct.item() * 100 / len(test_data)
    return accuracy

# Convert the tensor to image
def tensor_to_image(tensor):
    tensor = tensor.detach().cpu()
    tensor = tensor.view(28, 28)
    return tensor.numpy()

def main():
    
    #copywriting side
    st.markdown("Hi, welcome to my page!")
    st.markdown("In this page you can test and train Neural Networks Model Classifier. There are two models that you can try, which are Artificial Neural Network (ANN) and Convolutional Neural Network (CNN). Dataset that used is MNIST. You can also take a look of the dataset preview on the side bar. 	:arrow_left:")
    st.markdown("Then, if you want to try training and evaluating the model, you need to set the Input Size, Output Size, Learning Rate, and Epoch")

    # Default values
    input_size = 784
    output_size = 10
    train_batch_size = 100
    test_batch_size = 500
    epochs = 10
    model_trained = False
    
    
    #SIDEBAR
    st.sidebar.header("Dataset Preview")
    train_batch_size = st.sidebar.number_input("Train Batch Size", min_value=1, max_value=1000, value=train_batch_size)
    test_batch_size = st.sidebar.number_input("Test Batch Size", min_value=1, max_value=1000, value=test_batch_size)
    
    train_data, test_data = load_data()
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    for images,labels in train_loader: 
        break
    jmlh_label = st.sidebar.slider('Show the label :', 1, train_batch_size)
    st.sidebar.text_input('Labels: ', labels[:jmlh_label].numpy())
    im = make_grid(images[:jmlh_label], nrow=12)
    fig = plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    st.sidebar.pyplot(fig)

    #main page before choose model
    input_size = st.number_input("Input Size", min_value=1, max_value=1000, value=input_size)
    output_size = st.number_input("Output Size", min_value=1, max_value=100, value=output_size)
    learn_rate = st.number_input("Masukkan Learning Rate (x 10):", 0.01, 1.000)
    st.write('Learning Rate: ', learn_rate/10)
    epochs = st.number_input("Masukkan Epochs:", 1, 10)
    st.write('Epoch: ', epochs)

    st.divider()

    st.markdown("After fill out those parameter, here's you can choose which model you want")

    menu = ["ANN", "CNN"]
    choice = st.selectbox("Menu", menu)
    if choice == "ANN":
        layers=[]
        jmlh_hidlay = st.number_input("Masukkan Jumlah Hidden Layer: ", 1, 5)
        for i in range (jmlh_hidlay):
            n = st.number_input("Masukkan neuron dalam hidden layer "f"{i+1}", 1, 1000)
            layers.append(n)
        if st.button("Train and Evaluate") or model_trained:
            train_data, test_data = load_data()
            train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

            model = MultilayerPerceptron(input_size, output_size, layers)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

            train_losses, test_losses, train_correct, test_correct = train(
                model, train_loader, test_loader, criterion, optimizer, epochs
            )

            st.write("Training completed!")

            # Combine training and test losses in one graph
            loss_chart = plt.figure()
            plt.plot(train_losses, label="Training Loss")
            plt.plot(test_losses, label="Test Loss")
            plt.title("Loss at the end of each epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            st.pyplot(loss_chart)

            # Combine training and test accuracy in one graph
            acc_chart = plt.figure()
            plt.plot(train_correct, label="Training Accuracy")
            plt.plot(test_correct, label="Test Accuracy")
            plt.title("Accuracy at the end of each epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            st.pyplot(acc_chart)

            # Confusion matrix and test accuracy
            with torch.no_grad():
                model.eval()
                predictions = []
                ground_truths = []
                for batch_idx, (X_test, y_test) in enumerate(test_loader):
                    output = model(X_test.view(-1, 28*28))
                    _, predicted = torch.max(output, 1)
                    predictions.extend(predicted.tolist())
                    ground_truths.extend(y_test.tolist())

                cm = confusion_matrix(ground_truths, predictions)
                accuracy = np.trace(cm) / np.sum(cm) * 100

                st.write("Confusion Matrix:")
                st.write(cm)

                st.write("Test Accuracy: {:.2f}%".format(accuracy))

        model_trained = True

                       
    elif choice == "CNN":
        model = ConvolutionalNetwork()
        # Set random seed
        #torch.manual_seed(42)
        if st.button("Train and Evaluate"):
            train_data, test_data = load_data()
            train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
            
            train_losses, train_correct, test_losses, test_correct = train_model(model, epochs, train_batch_size, train_loader, test_loader, criterion, optimizer)
            accuracy = evaluate_model(model, test_loader, test_data)

            # Display the training and validation loss
            st.subheader("Training and Validation Loss")
            loss_chart = plt.figure()
            plt.plot(train_losses, label='Training Loss')
            plt.plot(test_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            st.pyplot(loss_chart)

            # Display the training and validation accuracy
            acc_chart = plt.figure()
            st.subheader("Training and Validation Accuracy")
            plt.plot([t/(train_batch_size*600) for t in train_correct], label='Training Accuracy')
            plt.plot([t/len(test_data) for t in test_correct], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            st.pyplot(acc_chart)

            # Display the confusion matrix
            with torch.no_grad():
                all_predicted = torch.tensor([])
                all_labels = torch.tensor([])
                for X_test, y_test in test_loader:
                    output = model(X_test)
                    predicted = torch.max(output.data, 1)[1]
                    all_predicted = torch.cat((all_predicted, predicted), dim=0)
                    all_labels = torch.cat((all_labels, y_test), dim=0)
                confusion = confusion_matrix(all_predicted.view(-1), all_labels.view(-1))
            st.subheader("Confusion Matrix")
            st.write(confusion)

            # Test the model with a random image
            st.subheader("Test Random Image")
            random_index = np.random.randint(len(test_data))
            random_image, random_label = test_data[random_index]
            random_image = random_image.unsqueeze(0)
            random_output = model(random_image)
            random_prediction = torch.max(random_output.data, 1)[1].item()

            # Display the random image and prediction
            col1, col2 = st.columns(2)
            col1.image(tensor_to_image(random_image), caption=f"True Label: {random_label}")
            col2.write(f"Predicted Label: {random_prediction}")

            # Display the overall accuracy
            st.subheader("Overall Accuracy")
            st.write(f"Accuracy: {accuracy:.2f}%")

    st.caption("Made with 😭 by @indahnaabila")
        

if __name__ == "__main__":
    main()