from fastapi import FastAPI, WebSocket, WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_personal_answer(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def division(self, division_shape : str, websocket: WebSocket):
        await websocket.send_text(division_shape)

    async def system_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel


class ChatBotCharacter:
    def __init__(self, character_id):
        self.character_id = character_id

    def f_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(f'{self.character_id}_chatbot', bos_token='</s>', eos_token='</s>', pad_token='<pad>')
        model = TFGPT2LMHeadModel.from_pretrained(f'{self.character_id}_chatbot')
        return tokenizer, model

    def f_InsetName(self, data):
        CharacterName = ''
        if self.character_id == 'Kimgu':
            CharacterName = '김구'
        elif self.character_id == 'MacArthur':
            CharacterName = '맥아더'
        elif self.character_id == 'Leejungseob':
            CharacterName = '이중섭 화가'

        InsetNameData = f'{CharacterName}는 {data}'
        return InsetNameData, CharacterName

