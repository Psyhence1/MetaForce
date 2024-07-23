import pandas as pd                                                         #to read csv file
from sklearn.model_selection import train_test_split                        #to split the data into training and testing set
from sklearn.feature_extraction.text import TfidfVectorizer                 #to convert text into vectors
from sklearn.linear_model import LogisticRegression                         #to train the model
from sklearn.metrics import classification_report, accuracy_score           #to evaluate the model
import matplotlib.pyplot as plt                                             #to plot the confusion matrix
import seaborn as sns                                                       #to plot the confusion matrix
from sklearn.preprocessing import LabelEncoder                              #to encode the target labels
import tensorflow as tf                                                     #For deep learning
import numpy as np                                                          #for linear algebra

data = pd.read_csv('Classification of Similes and Metaphors.csv')           #read the csv file

# Text Data Analysis
# Sentence length analysis
data['sentence_length'] = data['Sentence'].apply(len)                       #calculate the length of each sentence
plt.figure(figsize=(10, 6))                                                 #plot the histogram
sns.histplot(data, x='sentence_length', hue='Target', kde=True)
plt.title('Sentence Length Distribution by Target')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.show()

# Word clouds
from wordcloud import WordCloud                                             #to generate word clouds

simile_text = ' '.join(data[data['Target'] == 'Simile']['Sentence'])        #join all the sentences of the simile class
metaphor_text = ' '.join(data[data['Target'] == 'Metaphor']['Sentence'])    #join all the sentences of the metaphor class

plt.figure(figsize=(14, 7))                                                 #plot the word cloud for similes
wordcloud_simile = WordCloud(width=800, height=400, background_color='white').generate(simile_text)
plt.imshow(wordcloud_simile, interpolation='bilinear')
plt.title('Word Cloud for Similes')
plt.axis('off')
plt.show()

# Step 1: Vectorize the Text
vectorizer = TfidfVectorizer()                                              #create a TfidfVectorizer object
X = vectorizer.fit_transform(data['Sentence'])                              #vectorize the text data
y = data['Target']                                                          #get the target labels

# Step 2: Encode the target labels
label_encoder = LabelEncoder()                                              #create a LabelEncoder object
y = label_encoder.fit_transform(y)   

# Step 3: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    #split the data into training and testing sets

# Step 4: Train the Model
model = LogisticRegression()                                                #create a LogisticRegression object
model.fit(X_train, y_train)                                                 #train the model on the training data

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)                                              #make predictions on the testing data

print("Accuracy:", accuracy_score(y_test, y_pred))                          #print the accuracy score
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Import ML libnraries
from tensorflow.keras.models import Sequential                              #to create a sequential model
from tensorflow.keras.layers import Dense, Dropout                          #to add layers to the model
from tensorflow.keras.optimizers import Adam                                #to use Adam optimizer

# Step 7: Build the Neural Network Model
model = Sequential([                                                        #create a sequential model
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),         #add a dense layer with 512 units and ReLU activation function
    Dropout(0.5),                                                           #add a dropout layer with a dropout rate of 0.5
    Dense(256, activation='relu'),                                          #add another dense layer with 256 units and ReLU activation function
    Dropout(0.5),                                                           #add another dropout layer with a dropout rate of 0.5
    Dense(1, activation='sigmoid')                                          #add a dense layer with 1 unit and sigmoid activation function
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),                           #use Adam optimizer with a learning rate of 0.0001
              loss='binary_crossentropy',                                   #use binary cross-entropy loss function
              metrics=['accuracy'])                                         #evaluate the model on accuracy

# Print model summary
model.summary()

# Train the model 1st attempt...                                            !!!  #sparse tensor error
history = model.fit(X_train, y_train,
                    epochs=75,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    verbose=2)

# Assuming X_train and X_test are sparse tensors
X_train_reordered = tf.sparse.reorder(X_train)
X_test_reordered = tf.sparse.reorder(X_test)

# Train the model; 2nd attempt using reordered sparse tensors...            !!!  #csr_matrix error
history = model.fit(X_train_reordered, y_train,
                    epochs=75,
                    batch_size=64,
                    validation_data=(X_test_reordered, y_test),
                    verbose=2)

from scipy.sparse import csr_matrix                                         #for sparse matrix operations

# Convert csr_matrix to SparseTensor
X_train_sparse = tf.sparse.SparseTensor(indices=np.vstack((X_train.row, X_train.col)).T,
                                        values=X_train.data,
                                        dense_shape=X_train.shape)
X_test_sparse = tf.sparse.SparseTensor(indices=np.vstack((X_test.row, X_test.col)).T,
                                       values=X_test.data,
                                       dense_shape=X_test.shape)

# Reorder the SparseTensors
X_train_reordered = tf.sparse.reorder(X_train_sparse)
X_test_reordered = tf.sparse.reorder(X_test_sparse)

# Train the model; 3rd attempt using reordered sparse tensors after csr_matrix conversion...    !!!  #col/row attributes error
history = model.fit(X_train_reordered, y_train,
                    epochs=75,
                    batch_size=64,
                    validation_data=(X_test_reordered, y_test),
                    verbose=2)

# Convert csr_matrix to SparseTensor
X_train_sparse = tf.sparse.SparseTensor(indices=np.vstack((X_train.tocoo().row, X_train.tocoo().col)).T,
                                        values=X_train.data,
                                        dense_shape=X_train.shape)
X_test_sparse = tf.sparse.SparseTensor(indices=np.vstack((X_test.tocoo().row, X_test.tocoo().col)).T,
                                       values=X_test.data,
                                       dense_shape=X_test.shape)

# Reorder the SparseTensors
X_train_reordered = tf.sparse.reorder(X_train_sparse)
X_test_reordered = tf.sparse.reorder(X_test_sparse)

# Train the model; 4th attempt after constructing indexes for reordered converted sparse tensors...    !!! Sucess !!!                
history = model.fit(X_train_reordered, y_train,
                    epochs=75,
                    batch_size=64,
                    validation_data=(X_test_reordered, y_test),
                    verbose=2)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Accuracy (Training)')
plt.plot(history.history['val_accuracy'], label='Accuracy (Validation)')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
