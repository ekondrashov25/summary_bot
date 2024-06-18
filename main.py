import time
import logging
import threading
from datetime import datetime

import telebot
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from faster_whisper import WhisperModel
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


import config
from models import Message


mistral_model = config.mistral_model
whisper_model = config.whisper_model
group_id = config.group_id
tg_api_key = config.tg_api_key
mistral_api_key = config.mistral_api_key
max_messages = config.max_messages
eleven_labs_api_key = config.eleven_labs_api_key

temperature = config.model_temperature
max_tokens = config.max_tokens

inital_prompt = config.initial_prompt
delete_time = config.delete_message_time

format = config.format

logger = logging.getLogger(__name__)
logging.basicConfig(format=format, encoding='utf-8',level=logging.INFO)

bot = telebot.TeleBot(tg_api_key)
client = MistralClient(mistral_api_key)
whisper = WhisperModel(whisper_model, device="cpu", compute_type="int8")
eleven_labs_client = ElevenLabs(api_key=eleven_labs_api_key)

context = []
messages_to_delete = {}


@bot.message_handler(commands=['history'])
def restore_messages(message: telebot.types.Message):

    bot.send_message(group_id, 'Restoring messages, please wait...')
    str_dialogue = '\n'.join([f'({mess.timestamp}) {mess.user_name}: {mess.msg_text}' for mess in sorted(context, key=lambda x: x.timestamp)])
    logger.info(f'Restored dialogue: {str_dialogue}')

    messages = [ChatMessage(role="system", content=inital_prompt), ChatMessage(role='user', content=str_dialogue)]

    try:
        summary = client.chat(
            model=mistral_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        audio = eleven_labs_client.generate(
            text=summary.choices[0].message.content,
            voice="Rachel",
            model="eleven_multilingual_v2"
        )
        save(audio, "test.opus")
        
        # summary = bot.send_message(group_id, summary.choices[0].message.content)
        # threading.Thread(target=delete_message, args=(summary.message_id, summary.chat.id)).start()

        with open('test.opus', 'rb') as file:
            bot.send_audio(group_id, file)

    except Exception as e:
        logger.error(f'Error generating summary: {e}')
        bot.send_message(group_id, 'Sorry, I encountered an error while generating the summary')


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    logger.info(f'user send voice message')
    file_info = bot.get_file(message.voice.file_id)
    df = bot.download_file(file_info.file_path)

    with open(f'voice/{message.from_user.id}_{message.message_id}.mp3', 'wb+') as file:
        file.write(df)
        logger.info(f'voice writed to voice/{message.from_user.id}_{message.message_id}.mp3 ')

    logger.info(f'transcribing file using whisper ({whisper_model=})')

    segments, _ = whisper.transcribe(f"voice/{message.from_user.id}_{message.message_id}.mp3", beam_size=5)
    recognized_text = ''.join(segment.text for segment in segments)

    logger.info(f'successfully transcribed audio file: {recognized_text[:50]}... (first 50 chars)')

    message_time = datetime.fromtimestamp(message.date)
    context.append(Message(timestamp=message_time, user_name=message.from_user.full_name, msg_text=recognized_text))

    logger.info('successfully added recognised text to context')

@bot.message_handler(content_types=['video_note'])
def handle_video(message):
    logger.info('user send vide_note')
    file_id = message.video_note.file_id

    file_info = bot.get_file(file_id)
    file_path = file_info.file_path

    downloaded_file = bot.download_file(file_path)

    logger.info('successfully downloaded file')

    with open(f"voice_notes/{message.from_user.id}_{message.message_id}.mp3", 'wb+') as file:
        file.write(downloaded_file)

    logger.info(f'transcribing video_note using whisper ({whisper_model=})')

    segments, _ = whisper.transcribe(f"voice_notes/{message.from_user.id}_{message.message_id}.mp3", beam_size=5)
    recognized_text = ''.join(segment.text for segment in segments)

    logger.info(f'successfully transcribed video_not file: {recognized_text[:50]}... (first 50 chars)')

    message_time = datetime.fromtimestamp(message.date)
    context.append(Message(timestamp=message_time, user_name=message.from_user.full_name, msg_text=recognized_text))

    logger.info('successfully added recognized text to context')

@bot.message_handler(func=lambda message: True)
def main(message: telebot.types.Message):
    message_time = datetime.fromtimestamp(message.date)
    context.append(Message(timestamp=message_time, user_name=message.from_user.full_name, msg_text=message.text))

    if len(context) > max_messages:
        context.pop(0)
    

def delete_message(message_id, chat_id):
    time.sleep(60 * delete_time)
    try:
        bot.delete_message(chat_id, message_id)
    except Exception as e:
        logger.error(f"Error deleting message: {e}")

if __name__ == "__main__":
    logger.info('bot started')
    bot.infinity_polling()
