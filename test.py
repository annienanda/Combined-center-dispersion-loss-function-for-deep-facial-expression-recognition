from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, PReLU, Embedding, Lambda
from keras import backend as K
import numpy as np
import dataset

# Model Parameters
num_classes = 8
hidden_dim = 4096
batch_size = 32
img_size = 224
test_path = "/home/aimlab/Downloads/FERPlus-master/src/FERPLUS_TEST"

# Load Dataset
data = dataset.read_train_sets(test_path, img_size, classes=['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'], validation_size=0.0)
x_test = data.train.images
y_test = data.train.labels
y_test_value = np.argmax(y_test, axis=1).reshape(-1, 1)  # Convert labels to indices

# Load VGGFace Model
base_model = VGGFace(input_shape=(img_size, img_size, 3), include_top=True)
print("VGGFace Model Loaded.")

# Build Custom Classifier
last_layer = base_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, name='fc6')(x)
x = Activation('relu', name='fc6/relu')(x)
x = Dense(hidden_dim, name='fc7')(x)
x = Activation('relu', name='fc7/relu')(x)
ip1 = PReLU(name='ip1')(x)
ip2 = Dense(num_classes, name='fc8')(ip1)
ip2 = Activation('softmax', name='fc8/softmax')(ip2)

# CCDL Loss
input_target = Input(shape=(1,), name='target')
centers = Embedding(num_classes, hidden_dim)(input_target)
l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([ip1, centers])
eq_loss = Lambda(lambda x: 0.5 * K.mean(K.square(x[0][:, 0] - x[1][:, 0]), 1, keepdims=True), name='eq_loss')([centers, centers])

# Define Model
model_centerloss = Model(inputs=[base_model.input, input_target], outputs=[ip2, l2_loss, eq_loss])

# Freeze Initial Layers
for layer in model_centerloss.layers[:7]:
    layer.trainable = False

# Load and Evaluate 5 Snapshot Models
learning_rates = [0.013, 0.014, 0.015, 0.016, 0.017]
predictions = []

for i in range(5):
    model_centerloss.load_weights(f"snapshot_model_{i+1}.h5")
    print(f"Loaded Snapshot Model {i+1}.")

    # Predict Probabilities
    preds, _, _ = model_centerloss.predict([x_test, y_test_value], batch_size=batch_size)
    predictions.append(preds)

# Average the Predictions
final_predictions = np.mean(predictions, axis=0)

# Convert to Class Labels
final_labels = np.argmax(final_predictions, axis=1)

# Calculate Accuracy
accuracy = np.mean(final_labels == np.argmax(y_test, axis=1))
print(f"Final Ensemble Test Accuracy: {accuracy:.4f}")
