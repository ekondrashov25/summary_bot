import os
import time
import uuid
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
eleven_labs_model = config.eleven_labs_model
eleven_labs_voice = config.eleven_labs_voice

temperature = config.model_temperature
max_tokens = config.max_tokens

inital_prompt = config.initial_prompt
delete_time = config.delete_message_time

format = config.format

voice_path = f'{group_id}/voice'
voice_notes_path = f'{group_id}/voice_notes'
voice_summary_path = f'{group_id}/voice_summary'

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
    restore_message = bot.send_message(group_id, 'Restoring messages, please wait...')
    logger.info(f'user ({message.from_user.id}) requested a summary')

    str_dialogue = '\n'.join([f'({mess.timestamp}) {mess.user_name}: {mess.msg_text}' for mess in sorted(context, key=lambda x: x.timestamp)])
    logger.info(f'restored dialogue:\n{str_dialogue}')

    messages = [ChatMessage(role="system", content=inital_prompt), ChatMessage(role='user', content=str_dialogue)]

    logger.info(f'context with dialogue for model: {messages}')

    try:
        logger.info('creating text summary')

        summary = client.chat(
            model=mistral_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if len(message.text.split()) > 1 and message.text.split()[1] == 'audio':
            logger.info('user requested audio summary')
            try:
                audio = eleven_labs_client.generate(
                    text=summary.choices[0].message.content,
                    voice=eleven_labs_voice,
                    model=eleven_labs_model
                )   
                logger.info(f'successfully created audio summary')
                file_name =  f'{voice_summary_path}/{message.from_user.id}_{uuid.uuid4()}.ogg'

                save(audio, file_name)
                logger.info(f'file ({file_name}) successfully saved')

                with open(file_name, 'rb') as file:
                    time.sleep(0.5)
                    bot.delete_message(restore_message.chat.id, restore_message.message_id)
                    summary = bot.send_audio(group_id, file)

                    logger.info('send file and starting delete_message function')
                    threading.Thread(target=delete_message, args=(summary.message_id, summary.chat.id)).start()

            except Exception as e:
                logger.error(f'error while generating audio summary: {e}')
                bot.send_message(group_id, 'sorry, I encountered an error while generating the audio summary')
        else:
            logger.info("user don't required an audio summary")
            time.sleep(0.5)
            summary = bot.edit_message_text(summary.choices[0].message.content, restore_message.chat.id, restore_message.message_id)
            logger.info('send summary and starting delete function')

            threading.Thread(target=delete_message, args=(summary.message_id, summary.chat.id)).start()

    except Exception as e:
        logger.error(f'error generating summary: {e}')
        bot.send_message(group_id, 'sorry, I encountered an error while generating the summary')


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    logger.info(f'user send voice message')
    file_info = bot.get_file(message.voice.file_id)
    df = bot.download_file(file_info.file_path)


    file_name = f'{voice_path}/{message.from_user.id}_{message.message_id}.mp3'

    with open(file_name, 'wb+') as file:
        file.write(df)
        logger.info(f'voice writed to {file_name}')

    logger.info(f'transcribing file using whisper ({whisper_model=})')

    segments, _ = whisper.transcribe(file_name, beam_size=5)
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

    file_name = f'{voice_notes_path}/{message.from_user.id}_{message.message_id}.mp3'

    with open(file_name, 'wb+') as file:
        file.write(downloaded_file)

    logger.info(f'transcribing video_note using whisper ({whisper_model=})')

    segments, _ = whisper.transcribe(file_name, beam_size=5)
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
        logger.info('stack message limit exceeded')
        context.pop(0)
    

def delete_message(message_id: int, chat_id: int | str) -> None:
    logger.info(f'called function for delete, waitinig {60 * delete_time}')
    time.sleep(60 * delete_time)

    try:
        bot.delete_message(chat_id, message_id)
        logger.info(f'message ({message_id}) deleted from chat_id ({chat_id})')
    except Exception as e:
        logger.error(f"error deleting message: {e}")


def create_directories(dirs: list[str], group_id: int):
    for dir in dirs:
        if not os.path.exists(dir):
            logger.info(f'there is no correct directory, creating {dir}... ')
            os.makedirs(dir)
        else:
            logger.info(f'directory {dir} already exists')

if __name__ == "__main__":
    logger.info('creating directories')
    
    directories = [voice_path, voice_notes_path, voice_summary_path]
    create_directories(directories, group_id)

    logger.info('directories successfully created ')
    logger.info('bot started...')
    
    bot.infinity_polling()
    print('hello')
