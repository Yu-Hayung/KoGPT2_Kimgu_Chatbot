import asyncio
from datetime import datetime

from fastapi import FastAPI, WebSocket, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect
from fastapi.logger import logger
from enum import Enum, auto
from fastapi_utils.enums import StrEnum

from function import *
from apps import *

app = FastAPI()
# html파일을 서비스할 수 있는 jinja설정 (/templates 폴더사용)
templates = Jinja2Templates(directory="templates")

# 웹소켓 연결을 테스트 할 수 있는 웹페이지 (http://127.0.0.1:8000/client)



@app.get("/Main")
async def client(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/Kimgu")
async def client(request: Request):
    # /templates/client.html파일을 response함
    return templates.TemplateResponse("client_Kimgu.html", {"request": request})

@app.get("/MacArthur")
async def client(request: Request):
    return templates.TemplateResponse("client_MacArthur.html", {"request": request})


manager = ConnectionManager() #apps.py


# 웹소켓 설정 ws://127.0.0.1:8000/ws 로 접속할 수 있음
@app.websocket("/ws/{character_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int, character_id: str):
    await manager.connect(websocket) # client의 websocket접속 허용
    print(' *** 소켓 접속 완료 *** ')

    # transformers 미리 켜놓기
    from transformers import AutoTokenizer
    from transformers import TFGPT2LMHeadModel
    if character_id == 'Kimgu':
        tokenizer = AutoTokenizer.from_pretrained('Kimgu_chatbot', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
        model = TFGPT2LMHeadModel.from_pretrained('Kimgu_chatbot')
    elif character_id == 'MacArthur':
        tokenizer = AutoTokenizer.from_pretrained('MacArthur_chatbot', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
        model = TFGPT2LMHeadModel.from_pretrained('MacArthur_chatbot')

    data = '' # 지역변수 오류 방지

    try:
        while True:
            data = await websocket.receive_text()  # client 메시지 수신대기
            # print(f"message received : {data} from : {websocket.client}") # 상태 보기

            # AI의 답변 함수 [ answer ]
            if character_id == 'Kimgu':
                namedata = f'김구는 {data}'
            elif character_id == 'MacArthur':
                namedata = f'맥아더는 {data}'

            await manager.send_personal_message(f"| 사용자  : {data} ", websocket)  # client에 받은 메세지
            await manager.broadcast(f"'")
            answer = chatbot(namedata, tokenizer, model)                              # 자연어 분석
            await manager.send_personal_answer(f"| 김구  : {answer}", websocket)  # client에 메시지 전달
            await manager.broadcast(f'***' * 30)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.send_personal_answer(f"사용자 #{client_id} 님이 퇴장 했습니다.", websocket)
        await manager.broadcast(f" 사용자 #{client_id} 님이 퇴장 했습니다.")



# 개발/디버깅용으로 사용할 앱 구동 함수
def run():
    import uvicorn
    uvicorn.run(app)

# python main.py로 실행할경우 수행되는 구문
# uvicorn main:app 으로 실행할 경우 아래 구문은 수행되지 않는다.
if __name__ == "__main__":
    run()