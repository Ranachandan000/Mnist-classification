#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target


# * I'm taking 10k data points due to runtime issues in my pc.

# In[2]:


X = X[:10000]
y = y[:10000]                    


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### (a) Modify the code so that it uses L1-distance instead of the default L2-distance (Euleadean).

# In[4]:


knn_classifier = KNeighborsClassifier(n_neighbors=5 , p = 1)    # for p = 1 , metric is manhatten i.e. L1 distance
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ### (b) Find out the K that gives better performance.

# In[5]:


best_accuracy = 0.0
best_k = 0
for k in range(1, 11):                         # 10 iterations only
    knn = KNeighborsClassifier(n_neighbors=k, p = 1)
    knn.fit(X_train, y_train)
    X_test_pred = knn.predict(X_test)
    val_accuracy = accuracy_score(y_test, X_test_pred)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_k = k


# * Accuracy is calculated here for the test dataset

# In[6]:


print('''Best accuracy with "L1 distance" as the distance metric is =''', best_k)


# ### (c) Report the Accuracy

# In[7]:


final_knn = KNeighborsClassifier(n_neighbors=best_k, p = 1)                # p ==1 is manhatten distance == L1 distance
final_knn.fit(X_train, y_train)


# In[10]:


X_test_pred = final_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, X_test_pred)

print(f"Best K: {best_k}")
print(f"Test Accuracy with Best K: {test_accuracy}")


# ### (d) Display results by showing the image, actual label, and predicted label. Find out a few samples where the predicted labelis incorrect.

# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# * Displaying 5 samples where it has been predicted incorrectly

# In[19]:


incorrect_indices = []
for index, (true_label, predicted_label) in enumerate(zip(y_test, X_test_pred)):
    if true_label != predicted_label:
        incorrect_indices.append(index)


# In[21]:


print(len(incorrect_indices))                            # i.e. 111 samples were predicted incorrectly


# In[25]:


for index in incorrect_indices[:10]:                     # printing first 5 incorrect samples
    image = X_test[index, :].reshape(28, 28)
    correct_label = y_test[index]
    predicted_label = X_test_pred[index]

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray')
    plt.title(f"Correct Label: {correct_label},Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.show()


# In[ ]:




