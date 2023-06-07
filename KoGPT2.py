import time

import torch
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

# gpu 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print(' +++++ GPU 빌드 완료  +++++ ')


tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)


import pandas as pd
import tqdm

train_data = pd.read_csv('Data/KimguData.csv', encoding='utf-8')

print('train_data >>', len(train_data))


def chat_dataset():
    for question, answer in zip(train_data.Q.to_list(), train_data.A.to_list()):
        bos_token = [tokenizer.bos_token_id]
        eos_token = [tokenizer.eos_token_id]
        sent = tokenizer.encode('<usr>' + question + '<sys>' + answer)
        yield bos_token + sent + eos_token

batch_size = 32
dataset = tf.data.Dataset.from_generator(chat_dataset, output_types=tf.int32)
dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(None,), padding_values=tokenizer.pad_token_id)


adam = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
steps = len(train_data) // batch_size + 1
print('steps >>', steps)

EPOCHS = 50

for epoch in range(EPOCHS):
    epoch_loss = 0

    for batch in tqdm.tqdm(dataset,
                      total=steps,
                      desc='실행중',
                      initial=0,
                      leave=True,
                      ascii=True):
        with tf.GradientTape() as tape:
            result = model(batch, labels=batch)
            loss = result[0]
            batch_loss = tf.reduce_mean(loss)

        grads = tape.gradient(batch_loss, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += batch_loss / steps
        time.sleep(5)

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, epoch_loss))



# 전체 모델을 HDF5 파일로 저장
# model.save('Kimgu_model.h5')

tokenizer.save_pretrained('Kimgu_chatbot')
model.save_pretrained('Kimgu_chatbot')
