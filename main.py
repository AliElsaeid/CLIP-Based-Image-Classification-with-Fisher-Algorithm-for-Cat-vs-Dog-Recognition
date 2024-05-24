
# Step 2: Import necessary libraries
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from clip import load
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image  # Import the Image class

# Step 3: Load the dataset
df = pd.read_csv('cat_dog1.csv')

# Step 4: Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Step 5: Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = load("ViT-B/32", device)

# Step 6: Extract CLIP embeddings for the training set
train_embeddings = []
for img_name in train_df['image']:
    img_path = f"cat_dog1/{img_name}"  # Replace with your actual path
    image = transform(Image.open(img_path)).unsqueeze(0).to(device)
    embedding = model.encode_image(image)
    train_embeddings.append(embedding.cpu().detach().numpy())

# Convert the list to a numpy array
train_embeddings = np.vstack(train_embeddings)

# Step 7: Implement Fisher algorithm
C = 0.1
class_means = [train_embeddings[train_df['labels'] == label].mean(axis=0) for label in [0, 1]]
S_w = np.cov(train_embeddings[train_df['labels'] == 0], rowvar=False) + np.cov(train_embeddings[train_df['labels'] == 1], rowvar=False)
S_w_inv = np.linalg.inv(S_w)
w = C * np.dot(S_w_inv, (class_means[1] - class_means[0]))

# Step 8: Extract CLIP embeddings for the testing set
test_embeddings = []
for img_name in test_df['image']:
    img_path = f"cat_dog1/{img_name}"  # Replace with your actual path
    image = transform(Image.open(img_path)).unsqueeze(0).to(device)
    embedding = model.encode_image(image)
    test_embeddings.append(embedding.cpu().detach().numpy())

# Convert the list to a numpy array
test_embeddings = np.vstack(test_embeddings)

# Step 9: Apply Fisher algorithm to the testing set
predictions = np.dot(test_embeddings, w.T)
predicted_labels = (predictions > 0).astype(int)

# Step 10: Evaluate the model
conf_matrix = confusion_matrix(test_df['labels'], predicted_labels)
accuracy = accuracy_score(test_df['labels'], predicted_labels)
precision = precision_score(test_df['labels'], predicted_labels)
recall = recall_score(test_df['labels'], predicted_labels)
f1 = f1_score(test_df['labels'], predicted_labels)

# Step 11: Visualize results
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")