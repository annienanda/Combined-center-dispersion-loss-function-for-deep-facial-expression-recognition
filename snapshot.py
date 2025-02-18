from keras.callbacks import Callback
import numpy as np
import keras.backend as K

class SnapshotEnsemble(Callback):
    def __init__(self, n_epochs, n_cycles, lr_max):
        super(SnapshotEnsemble, self).__init__()
        self.n_epochs = n_epochs
        self.n_cycles = n_cycles
        self.lr_max = lr_max
        self.lr_schedule = []

    def on_train_begin(self, logs=None):
        for cycle in range(self.n_cycles):
            for epoch in range(self.n_epochs // self.n_cycles):
                cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch / (self.n_epochs // self.n_cycles))))
                lr = self.lr_max * cosine_decay
                self.lr_schedule.append(lr)

    def on_epoch_begin(self, epoch, logs=None):
        K.set_value(self.model.optimizer.lr, self.lr_schedule[epoch])
        print(f"Epoch {epoch + 1}: Learning Rate = {self.lr_schedule[epoch]:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % (self.n_epochs // self.n_cycles) == 0:
            print(f"Snapshot taken at epoch {epoch + 1}")

