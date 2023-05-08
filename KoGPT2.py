import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel


tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', from_pt=True)


import pandas as pd
import tqdm

train_data = pd.read_csv('Kimgu_data.csv', encoding='utf-8')
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

EPOCHS = 15

for epoch in range(EPOCHS):
    epoch_loss = 0

    for batch in tqdm.tqdm_notebook(dataset, total=steps):
        with tf.GradientTape() as tape:
            result = model(batch, labels=batch)
            loss = result[0]
            batch_loss = tf.reduce_mean(loss)

        grads = tape.gradient(batch_loss, model.trainable_variables)
        adam.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += batch_loss / steps

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, epoch_loss))



# 전체 모델을 HDF5 파일로 저장
# model.save('Kimgu_model.h5')

tokenizer.save_pretrained('Kimgu_chatbot')
model.save_pretrained('Kimgu_chatbot')


# tokenizer2 = AutoTokenizer.from_pretrained('Kimgu_chatbot', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
# model2 = TFGPT2LMHeadModel.from_pretrained('Kimgu_chatbot', from_pt=True)
#
# # test
# def return_answer_by_chatbot(user_text):
#     print('***'*20)
#     print('질문 >>', user_text)
#     sent = '<usr>' + user_text + '<sys>'
#     input_ids = [tokenizer2.bos_token_id] + tokenizer2.encode(sent)
#     input_ids = tf.convert_to_tensor([input_ids])
#     output = model2.generate(input_ids, max_length=50, do_sample=True, top_k=20)
#     sentence = tokenizer2.decode(output[0].numpy().tolist())
#     chatbot_response = sentence.split('<sys> ')[1].replace('</s>', '')
#     print('대답 >>', chatbot_response)
#     return chatbot_response
#
#
# return_answer_by_chatbot('별명이 있니')
# return_answer_by_chatbot('기차를 타봤니?')
# return_answer_by_chatbot('감옥 생활은 어때')
# return_answer_by_chatbot('백범 김구는 운동을 잘하니?')
# return_answer_by_chatbot('백범 김구는 커피를 좋아해요?')
# return_answer_by_chatbot('김구너는 인천에서 무엇을 했니?')
# return_answer_by_chatbot('인생의 목적이 무엇이니?')
# return_answer_by_chatbot('결혼했니?')
# return_answer_by_chatbot('아들이 있니?')
# return_answer_by_chatbot('초등학교를 설립한 적이 있니?')
# return_answer_by_chatbot('인천에서 무엇을 했니?')
# return_answer_by_chatbot('독립운동을 왜 했나요?')
