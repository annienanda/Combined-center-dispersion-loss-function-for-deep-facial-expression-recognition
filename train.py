from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Activation, PReLU, Embedding, Lambda
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import dataset
from snapshot import SnapshotEnsemble

# Model Parameters
num_classes = 8
hidden_dim = 4096
batch_size = 32
img_size = 224
epochs = 50
n_cycles = epochs // 10
train_path = "/home/aimlab/Downloads/FERPlus-master/src/FERPLUS_VAL"

# Load Dataset
data = dataset.read_train_sets(train_path, img_size, classes=['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt'], validation_size=0.0)
x_train = data.train.images
y_train = data.train.labels
y_train_value = np.argmax(y_train, axis=1).reshape(-1, 1)  # Convert labels to indices
random_y_train = np.random.rand(x_train.shape[0], 1)

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

# Compile Model
model_centerloss.compile(
    optimizer=SGD(lr=0.05),
    loss=['categorical_crossentropy', lambda y_true, y_pred: y_pred, lambda y_true, y_pred: y_pred],
    loss_weights=[1, 0.001, 0.001],
    metrics=['accuracy']
)

# Train 5 Different Models Using Snapshot Ensemble
learning_rates = [0.013, 0.014, 0.015, 0.016, 0.017]
for i, lr in enumerate(learning_rates):
    print(f"\nTraining Snapshot Model {i+1} with Learning Rate {lr}...")
    snapshot = SnapshotEnsemble(epochs, n_cycles, lr)
    model_centerloss.fit(
        [x_train, y_train_value], [y_train, random_y_train, random_y_train],
        epochs=10, batch_size=batch_size, callbacks=[snapshot], verbose=1
    )
    model_centerloss.save_weights(f"snapshot_model_{i+1}.h5")
    print(f"Snapshot Model {i+1} saved as 'snapshot_model_{i+1}.h5'.")

print("\nAll snapshot models trained and saved.")
