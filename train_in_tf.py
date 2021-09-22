import tensorflow as tf
from dataset import dataset, vocab_size, dataset_val
from evaluation import evaluator, beam_search

def loss_func(y_in, xy):
    y_mask = tf.cast(tf.expand_dims(tf.not_equal(y_in, 0), axis=2), tf.float32)
    y_in = y_in[:, 1:]
    y_mask = y_mask[:, 1:, 0]
    xy = xy[:, :-1]
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_in, xy
    )
    loss = tf.reduce_sum(loss * y_mask) / tf.reduce_sum(y_mask)
    return loss

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@tf.function(input_signature=train_step_signature)
def train_step(x_in, y_in):
    with tf.GradientTape() as tape:
        xy = model([x_in, y_in], None)
        loss = loss_func(y_in, xy)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

sentence = """abc"""
for epoch in range(epochs):
    for batch, ((x_in, y_in), _) in enumerate(dataset, start=1):
        loss = train_step(x_in, y_in)
        if batch % 50 == 0:
            print("loss:", loss)
            print(beam_search(model, sentence))
