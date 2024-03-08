# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow.keras as keras

[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
        "tf_flowers",
        split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
        as_supervised=True,
        with_info=True)

# %%
class_names=info.features["label"].names
n_classes=info.features["label"].num_classes
dataset_size=info.splits["train"].num_examples

# %%
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label


# %%
batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

# %%
plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()

# %%
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
    padding="same", activation="relu")

keras.layers.Rescaling(
    scale=1./255
)

# %%
model = keras.models.Sequential([
    keras.layers.Rescaling(scale=1./255),
    keras.layers.Conv2D(filters=32,kernel_size=7),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=64,kernel_size=5),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=128,kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=256,kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=256,kernel_size=3),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=5, activation='softmax'),
])



# %%
model.compile(optimizer='SGD',loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])

# %%
history = model.fit(train_set,validation_data=valid_set,epochs=10)

# %%
import pickle
a = (model.evaluate(train_set)[1],model.evaluate(valid_set)[1],model.evaluate(test_set)[1])
with open("simple_cnn_acc.pkl","wb") as f:
    pickle.dump(a,f)





