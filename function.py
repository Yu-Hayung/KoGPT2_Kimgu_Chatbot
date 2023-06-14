import tensorflow as tf
import re

def chatbot(text, tokenizer, model):
    # 불용어 단어 처리
    text = stop_words(text)
    if text == 'STOP_WORDS':
        return '지금 질문은 내가 대답 하기 어려운 질문을 했구나.'

    # input sentence : "질문" / 레이블 + 응답
    sentence = '<usr>' + text + '<sys>'
    tokenized = [tokenizer.bos_token_id] + tokenizer.encode(sentence)
    tokenized = tf.convert_to_tensor([tokenized])

    # 질문 문장으로 "레이블 + 응답" 토큰 생성
    output = model.generate(tokenized,
                            max_length=400,
                            do_sample=True,
                            top_k=4,
                            early_stopping=True
                            )
    sentence = tokenizer.decode(output[0].numpy().tolist())

    # 응답 토큰 생성
    response = sentence.split('<sys> ')[1].replace('</s>', '')
    response = hangul(response)

    #################################################

    print('***'*20)
    print(f'질문 : {text}')
    print(f'답변 : {response}')

    return response


def stop_words(InPutText:str):
    stop_word_list = ['담배', ' 술', '마약', '섹스', '성행위', '양주', '보드카'
                      ,'야동', 'AV', '포르노', '펜타닐', '맥주', '소주', '알콜',
                      '죽여', '살인', '칼로', '총으로', '폰', '전화번호', '살아',
                      '멍청이', '시발', '씨발', '병신', '바보', '미친', 'ㅄ',
                      'ㅂㅅ', '욕', '개같은', '개 같은']

    for stop_word in stop_word_list:
        if InPutText.find(f'{stop_word}') == -1:
            pass
        else:
            return 'STOP_WORDS'

    return InPutText

def hangul(in_text:str):

    hangul = re.compile('[^ ㄱ-ㅣ가-힣|0-9]+')
    out_text = hangul.sub('', in_text)

    return out_text


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
