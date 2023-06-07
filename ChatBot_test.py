import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel


tokenizer = AutoTokenizer.from_pretrained('Kimgu_chatbot', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
model = TFGPT2LMHeadModel.from_pretrained('Kimgu_chatbot')

# test
def chatbot(text):
    # input sentence : "질문" / 레이블 + 응답
    sentence = '<usr>' + text + '<sys>'
    tokenized = [tokenizer.bos_token_id] + tokenizer.encode(sentence)
    tokenized = tf.convert_to_tensor([tokenized])

    # 질문 문장으로 "레이블 + 응답" 토큰 생성
    output = model.generate(tokenized, max_length=50, do_sample=True, top_k=20)
    sentence = tokenizer.decode(output[0].numpy().tolist())
    # 응답 토큰 생성
    response = sentence.split('<sys> ')[1].replace('</s>', '')

    print('***' * 10, f' 서버 코드상 print 내용 HTML 화면과 다를 수 있습니다. ', '***' * 10)
    print(f'질문 : {text}')
    print(f'답변 : {response}')

    return response



chatbot('김구는 별명이 있니')
chatbot('김구는 기차를 타봤니?')
chatbot('김구는 감옥 생활은 어때')
chatbot('백범 김구는 운동을 잘하니?')
chatbot('백범 김구는 커피를 좋아해요?')
chatbot('김구 너는 인천에서 무엇을 했니?')
chatbot('김구는 인생의 목적이 무엇이니?')
chatbot('김구는 결혼했니?')
chatbot('김구는 아들이 있니?')
chatbot('김구는 초등학교를 설립한 적이 있니?')
chatbot('김구는 인천에서 무엇을 했니?')
chatbot('김구는 독립운동을 왜 했나요?')