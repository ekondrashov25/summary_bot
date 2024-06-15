from pydantic import BaseModel

from datetime import datetime

class Message(BaseModel):
    timestamp: datetime
    user_name: int | str
    msg_text: str

