import os
import numpy as np
import pandas as pd
import heapq
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, resize
from tensorflow.data import AUTOTUNE

# Define constants
DATA_DIR = 'Furniture_Data'
IMAGE_SIZE = (224, 224)
N_CATEGORIES = 6
N_STYLES = 17
BATCH_SIZE = 32
EPOCHS = 10

categories = ('beds', 'chairs', 'dressers', 'lamps', 'sofas', 'tables')

styles = ('Asian', 'Beach', 'Contemporary', 'Craftsman', 'Eclectic', 'Farmhouse', 'Industrial', 'Mediterranean', 
          'Midcentury', 'Modern', 'Rustic', 'Scandinavian', 'Southwestern', 'Traditional', 'Transitional', 'Tropical', 'Victorian')

# Dataframe of all images containing path, category and style columns
images_df = pd.DataFrame([(os.path.join(DATA_DIR, category, style, filename), category, style)
                          for category in os.listdir(DATA_DIR)
                          for style in os.listdir(os.path.join(DATA_DIR, category))
                          for filename in os.listdir(os.path.join(DATA_DIR, category, style))
                          if filename.endswith(('.jpg', '.jpeg'))],
                         columns=['path', 'category', 'style'])

print(images_df['style'].value_counts())
print()
print(images_df['category'].value_counts())
print()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)

# Load models if they exist, otherwise train them
if os.path.exists('models/category_classifier_cnn.h5'):
    category_classifier_model = load_model('models/category_classifier_cnn.h5')
else:
    # CNN model to classify furniture categories
    category_classifier_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(N_CATEGORIES, activation='softmax')
    ])
    # Split data into training and validation sets
    train_df, val_df = train_test_split(images_df, test_size=0.2, stratify=images_df['category'], random_state=1)
    
    datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Training data generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='category',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=1
    )
    # Validation data generator
    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='category',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=1
    )
 
    category_classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    category_classifier_model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=[early_stopping])
    category_classifier_model.save('models/category_classifier_cnn.h5')
    
category_classifier_model.summary()

# Autoencoder model to identify similar images
if os.path.exists('models/autoencoder.h5'):
    autoencoder = load_model('models/autoencoder.h5')
    autoencoder.summary()
else:
    # Trainging data generator
    data_generator = ImageDataGenerator(rescale=1.0/255).flow_from_dataframe(
        dataframe=images_df,
        x_col='path',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='input',
        seed=1
    )
    # Encoder
    encoder = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same')
    ])
    # Decoder
    decoder = Sequential([
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    # Autoencoder
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data_generator, epochs=EPOCHS)
    autoencoder.save('models/autoencoder.h5')
    
autoencoder.summary()

# CNN model to classify furniture styles
if os.path.exists('models/style_classifier_cnn.h5'):
    style_classifier_model = load_model('models/style_classifier_cnn.h5')
else:
    style_classifier_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(N_STYLES, activation='softmax')
    ])
    # Split data into training and validation sets
    train_df, val_df = train_test_split(images_df, test_size=0.2, stratify=images_df['style'], random_state=1)
    
    datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Trainging data generator
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='style',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=1
    )
    # Validation data generator
    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='style',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=1
    )

    style_classifier_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    style_classifier_model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator, callbacks=[early_stopping])
    style_classifier_model.save('models/style_classifier_cnn.h5')

style_classifier_model.summary()

######################### GUI ##############################

selected_img_path = None
selected_img_arr = None
category_pred = None
style_pred = None

def read_image(path, expand_dims=False):
    img = resize(decode_jpeg(read_file(path), channels=3), IMAGE_SIZE)
    if expand_dims:
        return tf.expand_dims(img, axis=0) / 255.0
    return img / 255.0

# Function to display images in a grid
def display_images(rows, cols, title, image_paths):
    fig, axs = plt.subplots(rows, cols, figsize=(12, 6))
    fig.suptitle(title)
    for i, image_path in enumerate(image_paths):
        row, col = i // cols, i % cols
        img = plt.imread(image_path)
        axs[row, col].imshow(img)
        axs[row, col].axis('off')
    plt.tight_layout()
    plt.show()

# Function to handle 'Select Image' button click event
def select_image():
    global selected_img_path
    opened_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg;*.jpeg')])
    if opened_path:
        selected_img_path = opened_path
        img = Image.open(selected_img_path).resize((250, 250))
        img = ImageTk.PhotoImage(img)
        path_entry.delete(0, tk.END)
        path_entry.insert(0, selected_img_path)
        image_label.config(image=img)
        image_label.image = img
        predict()

# Function to predict the category and style of the selected image
def predict():
    global selected_img_arr, category_pred, style_pred
    if selected_img_path:
        selected_img_arr = read_image(selected_img_path, True)
        category_pred = categories[np.argmax(category_classifier_model(selected_img_arr))]
        style_pred = styles[np.argmax(style_classifier_model(selected_img_arr))]
        result_label.config(text=f'Predicted Category: {category_pred}, Predicted Style: {style_pred}')

# Function to handle 'Find Similar' button click event
def find_similar():
    if category_pred and style_pred and selected_img_path:
        selected_image_encoded = np.array(autoencoder(selected_img_arr)).flatten()

        # Create a dataset to store image data, filtered by the predicted category and style
        batch_size = 32
        image_paths = images_df[(images_df['category'] == category_pred) & (images_df['style'] == style_pred)]['path'].values
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(read_image, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

        # Store top similarities in a priority queue
        similarities = []
        i = 0
        for batch in dataset:
            encoded_images = autoencoder(batch)
            for encoded_image in encoded_images:
                # Calculate similarity between two images based on Euclidean distance
                similarity = -np.linalg.norm(selected_image_encoded - np.array(encoded_image).flatten())
                # Keep only top 10 similarities
                if len(similarities) < 10:
                    heapq.heappush(similarities, (similarity, image_paths[i]))
                else:
                    heapq.heappushpop(similarities, (similarity, image_paths[i]))
                i += 1

        # Sort similar images by similarity in descending order
        similar_images = [images for _, images in sorted(similarities, key=lambda x: x[0], reverse=True)]

        # Display top 10 similar images
        display_images(2, 5, 'Top 10 Similar Images', similar_images)
    else:
        messagebox.showwarning('Info', 'Please select an image.')

# Function to handle Enter key press event in the text field
def on_enter(event):
    global selected_img_path
    path = path_entry.get().strip()
    if path:
        if os.path.exists(path):
            selected_img_path = path
            img = Image.open(selected_img_path).resize((250, 250))
            img = ImageTk.PhotoImage(img)
            path_entry.delete(0, tk.END)
            path_entry.insert(0, selected_img_path)
            image_label.config(image=img)
            image_label.image = img
            predict()
        else:
            messagebox.showwarning('Error', 'Invalid file path.')

# Create the main window
root = tk.Tk()
root.title('Model Testing GUI')

# Top buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, pady=10)
select_image_button = tk.Button(button_frame, text='Select Image', command=select_image)
select_image_button.pack(side=tk.LEFT, padx=5)
find_similar_button = tk.Button(button_frame, text='Find Similar', command=find_similar)
find_similar_button.pack(side=tk.LEFT, padx=5)

# Text field
path_frame = tk.Frame(root)
path_frame.pack()
path_label = tk.Label(path_frame, text='Path:')
path_label.pack(side=tk.LEFT)
path_entry = tk.Entry(path_frame, width=60)
path_entry.pack(side=tk.LEFT)
path_entry.bind('<Return>', on_enter)

# Image
image_label = tk.Label(root)
image_label.pack()

# Result text
result_label = tk.Label(root, text='')
result_label.pack(pady=10)

root.mainloop()
