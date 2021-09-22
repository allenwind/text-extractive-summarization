import tensorflow as tf

# 指定GPU并设置memory growth
physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from layers import *
from dataset import vocab_size

hdims = 256

# content和title输入
x_in = Input(shape=(None,), dtype=tf.int32, name="content") # content
y_in = Input(shape=(None,), dtype=tf.int32, name="title") # title
x = x_in
y = y_in

# 计算全局掩码
masking = GlobalMasking()

embedding = Embedding(
    input_dim=vocab_size,
    output_dim=hdims,
    embeddings_initializer="uniform"
)

x_mask = masking(x)
x_prior = PriorScaleShift(vocab_size)(x, mask=x_mask)

x = embedding(x)
y = embedding(y)

# encoder：Bi-LSTM对content进行编码
x = LayerNormalization()(x)
x = MaskBiLSTM(hdims // 2)(x, mask=x_mask)

x = LayerNormalization()(x)
x = MaskBiLSTM(hdims // 2)(x, mask=x_mask)

x = LayerNormalization()(x)
x = MaskBiLSTM(hdims // 2)(x, mask=x_mask)

# 聚合x的信息
x_pooling = MaskGlobalMaxPooling1D()(x, mask=x_mask)

# decoder：LSTM对title编码
y = ConditionalLayerNormalization(hdims // 4)([y, x_pooling])
y = LSTM(hdims, return_sequences=True)(y)

y = ConditionalLayerNormalization(hdims // 4)([y, x_pooling])
y = LSTM(hdims, return_sequences=True)(y)

y = ConditionalLayerNormalization(hdims // 4)([y, x_pooling])

# Attention交互1
xy = Attention(8, 16)([y, x, x], mask=x_mask)
# Concatenate融合
xy = Concatenate(axis=-1)([y, xy])
xy = Dense(hdims * 2)(xy)

# Attention交互2
# xy = Attention(8, 16)([xy, x, x], mask=x_mask)
# Concatenate融合
# xy = Concatenate(axis=-1)([y, xy])
# xy = Dense(hdims * 2)(xy)

xy = LeakyReLU(0.1)(xy)
xy = Dense(vocab_size)(xy)
xy = LayerNormalization()(xy)
alpha = 0.5
xy = Lambda(lambda x: alpha * x[0] + (1 - alpha) * x[1])([xy, x_prior])
xy = Activation("softmax")(xy)
xy = SparseCrossEntropy()([y_in, xy])
model = Model([x_in, y_in], xy)
model.summary()

lr = PiecewiseConstantDecay(
    boundaries=[1, 8, 50],
    values=[1e-2, 5*1e-3, 1e-3, 0.5*1e-3]
)
optimizer = Adam(lr)
model.compile(optimizer="adam")

train_code = 65
file = f"weights/weights.model-{train_code}"

if __name__ == "__main__":
    from dataset import dataset, dataset_val
    from dataset import test_files, tokenizer
    from dataset import train_files
    from evaluation import Evaluator
    from evaluation import load_test_sentences

    epochs = 10
    steps_per_epoch = int(len(train_files) / 128)
    # steps_per_epoch = 1000
    evaluator = Evaluator(load_test_sentences, tokenizer, file)
    callbacks = [evaluator]
    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=dataset_val,
        validation_batch_size=128,
        callbacks=callbacks
    )
    model.save_weights(file)
else:
    model.load_weights(file)
