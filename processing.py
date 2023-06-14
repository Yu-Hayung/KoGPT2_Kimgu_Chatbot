import os, re
import pandas as pd

##################################################################################
import openai
import os

# OpenAI API 키 설정
openai.api_key = "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" #민경범 대표님 key

def WebChatGPT(inputText):
    # ChatGPT에 입력 문장 전달
    response = openai.Completion.create(
        # engine="davinci",
        engine='text-davinci-003',
        prompt=inputText,
        max_tokens=4000,
        n=1,
        top_p=1,
        stop=None,
        temperature=0.5,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # OUT TEXT = response.choices[0].text.strip() Chat 답변

    return response, response.choices[0].text.strip()

##################################################################################

file_data = pd.read_csv('./Data/LeeJungSeobData.csv', encoding='utf-8')
setdata = file_data.drop_duplicates(subset='A')
Q_list = setdata['Q'].tolist()

num = 0
for i in Q_list:
    response, output = WebChatGPT(f'{i} \n 위 질문의 답변을 원하는게 아니라 위 질문을 다양한 '
                                  f'용언을 활용하여 질문 python 리스트 형태로 5개 만들어줘')

    Adata = str(setdata.loc[setdata['Q'] == f'{i}']['A'])
    hangul = re.compile('[^ 0-9|ㄱ-ㅣ가-힣]+')
    Adata = hangul.sub('', Adata)
    Adata = Adata.strip()

    for j in output:
        new_data = {'Q':[ f'{j}'], 'A': [f'{Adata}']}
        new_df = pd.DataFrame(new_data)
        file_data = pd.concat([file_data, new_df])
        print(f'{j} , {Adata}')

    if num == 2:
        break

    num += 1

print(file_data)
file_data.to_csv('./Data/Processing_test.csv', header=False, index=False)

