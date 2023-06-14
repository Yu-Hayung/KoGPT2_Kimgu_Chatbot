import asyncio
from datetime import datetime

from fastapi import FastAPI, WebSocket, Request, status
from fastapi.templating import Jinja2Templates

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

@app.get("/Leejungseob")
async def client(request: Request):
    return templates.TemplateResponse("client_Leejungseob.html", {"request": request})


manager = ConnectionManager() #apps.py



# 웹소켓 설정 ws://127.0.0.1:8000/ws 로 접속할 수 있음
@app.websocket("/ws/{character_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, character_id: str):
    await manager.connect(websocket) # client의 websocket접속 허용
    print(' *** 소켓 접속 완료 *** ')

    CBC = ChatBotCharacter(character_id)
    tokenizer, model = CBC.f_tokenizer()

    try:
        while True:
            data = await websocket.receive_text()  # client 메시지 수신대기
            # print(f"message received : {data} from : {websocket.client}") # 상태 보기

            # AI의 답변 함수 [ answer ]
            Insetnamedata, character_name = CBC.f_InsetName(data)

            await manager.send_personal_message(f"| 사용자  : {data} ", websocket)  # client에 받은 메세지
            await manager.broadcast(f"'")
            answer = chatbot(Insetnamedata, tokenizer, model)                              # 자연어 분석
            await manager.send_personal_answer(f"|  {character_name} : {answer}", websocket)  # client에 메시지 전달
            await manager.broadcast(f'***' * 30)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# 개발/디버깅용으로 사용할 앱 구동 함수
def run():
    import uvicorn
    uvicorn.run(app)

# python main.py로 실행할경우 수행되는 구문
# uvicorn main:app 으로 실행할 경우 아래 구문은 수행되지 않는다.
if __name__ == "__main__":
    run()