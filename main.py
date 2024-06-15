import telebot

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from datetime import datetime

from pydantic import BaseModel

import config 
from models import Message

bot = telebot.TeleBot(config.tg_api_key)
client = MistralClient(config.mistral_api_key)

model = config.mistral_model

print('bot started')

context = {}


@bot.message_handler(commands=['history'])
def restore_messages(message: telebot.types.Message):
    bot.send_message(config.group_id, 'restoring messages, wait a second...')
    dialogue = ''

    for user_id in context.keys():
        for mess in context[user_id]:
            dialogue += f'{mess.user_name}: {mess.msg_text}\n'

    messages = [ChatMessage(role="system", content=config.initial_prompt), ChatMessage(role='user', content=dialogue)]

    summary = client.chat(
    model=model,
    messages=messages,
    )

    bot.send_message(config.group_id, summary.choices[0].message.content)


@bot.message_handler(func=lambda message: True)
def main(message: telebot.types.Message):
    if message.from_user.id not in context.keys():
        context[message.from_user.id] = []

    message_time = datetime.fromtimestamp(message.date)
    context[message.from_user.id].append(Message(timestamp=message_time, user_name=message.from_user.full_name, msg_text=message.text))


bot.infinity_polling()