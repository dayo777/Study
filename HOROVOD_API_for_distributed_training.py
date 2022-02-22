# distributed training using HOROVOD api on Databricks
from sparkdl import HorovodRunner
from gc import callbacks
from tkinter.tix import Tree
import numpy as np
from cgi import test
import tensorflow as tf
import horovod.tensorflow.keras as hvd
from datetime import datetime

# assume we have 2 GPU clusters
strategy = tf.distribute.MirroredStrategy()


strategy.num_replicas_in_sync # to check no of GPUs

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

MODEL_NAME = 'myCIFAR10-{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))

# sychronize initial state of weights & biases amongst workers
hvd.callbacks.BroadcastGlobalVariablesCallback(0)

# average error metrics
hvd.callbacks.MetricAverageCallback()

# model CheckPoint as a Callback
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoint_path',
        monitor='val_accuracy', mode='max', save_best_only=True
    ))


def get_dataset(num_classes, rank=0, size=1):
    from tensorflow.keras import backend as K
    from tensorflow.keras import datasets, layers, models
    from tensorflow.keras.models import Sequential
    import tensorflow as tf
    from tensorflow import keras
    import horovod.tensorflow.keras as hvd
    import numpy as np

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    # 50000 train samples, 10000 test samples
    train_images = train_images[rank::size]
    train_labels = train_labels[rank::size]

    test_images = test_images[rank::size]
    test_labels = test_labels[rank::size]
    
    # normalize Pixel values
    train_images, test_images = train_images / 255.0, test_images / 255.0

    return train_images, train_labels, test_images, test_labels


def get_model(num_classes):
    from tensorflow.keras import models, layers, Sequential
    model = Sequential([
        layers.Conv2D(32, 3, activation='relu', name='conv_1', kernel_initializer='glorot_uniform',
                            padding='same', input_shape=(32, 32,3)),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, activation='relu', name='conv_2', kernel_initializer='glorot_uniform', padding='same'),
        layers.MaxPool2D(2),
        layers.Flatten(name='flat_1'),
        layers.Dense(256, activation='relu', kernel_initializer='glorot_initializer', name='dense_64'),
        layers.Dense(num_classes, activation='softmax', name='custom_class')
    ])
    model.build([None, 32, 32, 3])
    return model


def train_hvd(checkpoint_path, learning_rate=1.0):
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Sequential
    import tensorflow as tf
    from tensorflow import keras
    import horovod.tensorflow.keras as hvd
    import numpy as np

    # initialize Horovod
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')



# call the get_dataset function
num_classes = 10
train_images, train_labels, test_images, test_labels = get_dataset(num_classes, hvd.rank(), hvd.size())
validation_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

NUM_CLASSES = len(np.unique(train_labels))
BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
validation_dataset_size = len(test_labels)
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * hvd.size()
train_dataset = train_ds.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
validation_dataset = validation_ds.repeat().shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True
)

TRAIN_DATASET_SIZE = len(train_labels)
validation_dataset_size = len(test_labels)

STEPS_PER_EPOCH = TRAIN_DATASET_SIZE // BATCH_SIZE_PER_REPLICA
VALIDATION_STEPS = validation_dataset_size // BATCH_SIZE_PER_REPLICA
EPOCHS = 20


# create a model using the Get_model function
model = get_model(10)

# adjust the LR baseed on the no of GPUs
optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01*hvd.size())

# use the Horovod Distributed optimizer
optimizer = hvd.DistributedOptimizer(optimizer)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True
), metrics=['accuracy'])


callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)
]

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint_path',
    monitor='val_accuracy',
    mode='max',
    save_best_only = True))

model.fit(train_dataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=callbacks,
        validation_data=validation_dataset,
        validation_steps=VALIDATION_STEPS,
        verbose=1
)


# 2 processes per step
hr = HorovodRunner(np=2)
hr.run(train_hvd, checkpoint_path=checkpoint_path, learning_rate=0.01)
