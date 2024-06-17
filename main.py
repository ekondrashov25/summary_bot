import logging
import os
from datetime import datetime

import telebot
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


import config
from models import Message


logger = logging.getLogger(__name__)
logging.basicConfig(format=config.format, encoding='utf-8',level=logging.INFO)

model = config.mistral_model
group_id = config.group_id
tg_api_key = config.tg_api_key
mistral_api_key = config.mistral_api_key
max_messages = config.max_messages

temperature = config.model_temperature
max_tokens = config.max_tokens

inital_prompt = config.initial_prompt

bot = telebot.TeleBot(tg_api_key)
client = MistralClient(mistral_api_key)

logger.info('bot started')

context = []

@bot.message_handler(commands=['history'])
def restore_messages(message: telebot.types.Message):

    bot.send_message(group_id, 'Restoring messages, please wait...')
    str_dialogue = '\n'.join([f'({mess.timestamp}) {mess.user_name}: {mess.msg_text}' for mess in sorted(context, key=lambda x: x.timestamp)])
    logger.info(f'Restored dialogue: {str_dialogue}')


    messages = [ChatMessage(role="system", content=inital_prompt), ChatMessage(role='user', content=str_dialogue)]

    try:
        summary = client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        bot.send_message(group_id, summary.choices[0].message.content)
    except Exception as e:
        logger.error(f'Error generating summary: {e}')
        bot.send_message(group_id, 'Sorry, I encountered an error while generating the summary.')


@bot.message_handler(func=lambda message: True)
def main(message: telebot.types.Message):
    message_time = datetime.fromtimestamp(message.date)
    context.append(Message(timestamp=message_time, user_name=message.from_user.full_name, msg_text=message.text))


    if len(context) > max_messages:
        context.pop(0)
    

bot.infinity_polling()