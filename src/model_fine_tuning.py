import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

class VideoRecommendationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FineTunedCLIP(nn.Module):
    def __init__(self):
        super(FineTunedCLIP, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fc = nn.Linear(512, 10)  # Assume 10 classes for classification
        
    def forward(self, x):
        x = self.clip_model.get_image_features(x)
        x = self.fc(x)
        return x

def train_model(model, dataloader, epochs=5, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

def main():
    # Load your data here
    features = np.random.rand(100, 512)  # Simulated features
    labels = np.random.randint(0, 10, 100)  # Simulated labels
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    train_dataset = VideoRecommendationDataset(X_train, y_train)
    test_dataset = VideoRecommendationDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = FineTunedCLIP()
    train_model(model, train_loader)
    
    model.eval()
    with torch.no_grad():
        test_features = torch.tensor(X_test, dtype=torch.float32)
        test_labels = torch.tensor(y_test, dtype=torch.long)
        outputs = model(test_features)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test, predicted.numpy())
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()