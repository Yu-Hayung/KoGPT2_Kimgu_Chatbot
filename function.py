import tensorflow as tf

def chatbot(text, tokenizer, model):
    # input sentence : "질문" / 레이블 + 응답
    sentence = '<usr>' + text + '<sys>'
    tokenized = [tokenizer.bos_token_id] + tokenizer.encode(sentence)
    tokenized = tf.convert_to_tensor([tokenized])

    # 질문 문장으로 "레이블 + 응답" 토큰 생성
    output = model.generate(tokenized,
                            max_length=50,
                            do_sample=True,
                            top_k=20,
                            early_stopping=True
                            )
    sentence = tokenizer.decode(output[0].numpy().tolist())
    # 응답 토큰 생성
    response = sentence.split('<sys> ')[1].replace('</s>', '')

    #################################################

    print('***'*20)
    print(f'질문 : {text}')
    print(f'답변 : {response}')

    return response


# 파일 단독 실행시 활성화
# from transformers import AutoTokenizer
# from transformers import TFGPT2LMHeadModel
#
# tokenizer = AutoTokenizer.from_pretrained('Kimgu_chatbot', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
# model = TFGPT2LMHeadModel.from_pretrained('Kimgu_chatbot')
#
# chatbot('김구는 지하철 타봤나요?', tokenizer, model)
# chatbot('김구는 제트기 타봤나요?', tokenizer, model)
# chatbot('김구는 김밥을 좋아하나요?', tokenizer, model)
# chatbot('김구는 아내가 있나요?', tokenizer, model)
# chatbot('김구는 여동생이 있나요?', tokenizer, model)
