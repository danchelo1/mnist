import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.makedirs("models", exist_ok=True)

(x_train_all, y_train_all), (x_test, y_test) = keras.datasets.mnist.load_data()
x_all = np.concatenate([x_train_all, x_test], axis=0)
y_all = np.concatenate([y_train_all, y_test], axis=0)

x_all = x_all.astype("float32") / 255.0
x_all = np.expand_dims(x_all, -1)  # (N,28,28,1)

n = x_all.shape[0]
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
x_all = x_all[idx]
y_all = y_all[idx]

mid = n // 2
xA, yA = x_all[:mid], y_all[:mid]
xB, yB = x_all[mid:], y_all[mid:]


def split_data(x, y, test_size=0.1, val_size=0.1):
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=(test_size + val_size), stratify=y,
                                                        random_state=42)

    val_ratio = val_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=1 - val_ratio, stratify=y_temp,
                                                    random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


xA_train, yA_train, xA_val, yA_val, xA_test, yA_test = split_data(xA, yA, test_size=0.05, val_size=0.05)
xB_train, yB_train, xB_val, yB_val, xB_test, yB_test = split_data(xB, yB, test_size=0.05, val_size=0.05)

print("Shapes A train/val/test:", xA_train.shape, xA_val.shape, xA_test.shape)
print("Shapes B train/val/test:", xB_train.shape, xB_val.shape, xB_test.shape)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    shear_range=0.05,
    fill_mode='nearest'
)


def build_model_simple(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="simple_cnn")
    return model


def build_model_deep(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs, name="deep_cnn")
    return model


batch_size = 128
epochs = 12

# Model 1
model1 = build_model_simple()
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model1.summary())

history1 = model1.fit(
    xA_train, yA_train,
    validation_data=(xA_val, yA_val),
    epochs=epochs,
    batch_size=batch_size
)
model1.save("models/model_mnist.h5")

model2 = build_model_deep()
model2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
print(model2.summary())

train_generator = datagen.flow(xB_train, yB_train, batch_size=batch_size, shuffle=True, seed=42)
steps_per_epoch = max(1, len(xB_train) // batch_size)

history2 = model2.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=(xB_val, yB_val),
    epochs=epochs
)
model2.save("models/model_augmented.h5")

print("Training finished. Models saved in models/")

np.save("models/meta_test_A.npy", xA_test)
np.save("models/meta_test_yA.npy", yA_test)
np.save("models/meta_test_B.npy", xB_test)
np.save("models/meta_test_yB.npy", yB_test)
