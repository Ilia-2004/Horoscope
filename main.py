# подключение библиотек для бота
import telebot
import torch
import warnings
# подключение библиотек для языкового модуля
from telebot import types
from transformers import AutoTokenizer, AutoModelWithLMHead

# подключение бота
bot = telebot.TeleBot('6207624948:AAH9dI7o0JmTuZSm0az89jksRP9sYJbizoU')
# подключение языкового модуля
tokenizer = AutoTokenizer.from_pretrained('shahp7575/gpt2-horoscopes')
model = AutoModelWithLMHead.from_pretrained('shahp7575/gpt2-horoscopes')

 # Метод, получающий знак зодиака для языкового модуля
def make_prompt(category):
     return f"<|category|> {category} <|horoscope|>"

# Метод, который получает сообщения и обрабатывает их
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "Привет":
        # Пишем приветствие
        bot.send_message(message.from_user.id,
                         "Привет, {0.first_name}! \n Cейчас я расскажу тебе гороскоп на сегодня.".format(
                             message.from_user))
        # Готовим кнопки
        keyboard = types.InlineKeyboardMarkup()
        # По очереди готовим текст и обработчик для каждого знака зодиака
        key_oven = types.InlineKeyboardButton(text='♈️ Овен ♈️', callback_data='aries')
        keyboard.add(key_oven)

        key_telec = types.InlineKeyboardButton(text='♉️ Телец ♉️', callback_data='taurus')
        keyboard.add(key_telec)

        key_bliznecy = types.InlineKeyboardButton(text='♊️ Близнецы ♊️', callback_data='gemini')
        keyboard.add(key_bliznecy)

        key_rak = types.InlineKeyboardButton(text='♋️ Рак ♋️', callback_data='cancer')
        keyboard.add(key_rak)

        key_lev = types.InlineKeyboardButton(text='♌️ Лев ♌️', callback_data='leo')
        keyboard.add(key_lev)

        key_deva = types.InlineKeyboardButton(text='♍️ Дева ♍️', callback_data='virgo')
        keyboard.add(key_deva)

        key_vesy = types.InlineKeyboardButton(text='♎️ Весы ♎️', callback_data='libra')
        keyboard.add(key_vesy)

        key_scorpion = types.InlineKeyboardButton(text='♏️ Скорпион ♏️', callback_data='scorpio')
        keyboard.add(key_scorpion)

        key_strelec = types.InlineKeyboardButton(text='♐️ Стрелец ♐️', callback_data='sagittarius')
        keyboard.add(key_strelec)

        key_kozerog = types.InlineKeyboardButton(text='♑️ Козерог ♑️', callback_data='capricorn')
        keyboard.add(key_kozerog)

        key_vodoley = types.InlineKeyboardButton(text='♒️ Водолей ♒️', callback_data='aquarius')
        keyboard.add(key_vodoley)

        key_ryby = types.InlineKeyboardButton(text='♓️ Рыбы ♓️', callback_data='pisces')
        keyboard.add(key_ryby)
        # Показываем все кнопки сразу и пишем сообщение о выборе
        bot.send_message(message.from_user.id, text='Выбери свой знак зодиака', reply_markup=keyboard)

        return keyboard
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши Привет")
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")


@bot.message_handler(content_types=['text'])

# Обработчик нажатий на кнопки
@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    # Если нажали на одну из 12 кнопок — выводим гороскоп
    if call.data is not None:
        # задействование языкового модуля
        prompt = make_prompt(call.data)
        prompt_encoded = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
        sample_output = model.generate(prompt_encoded,
                                       do_sample=True,
                                        top_k=40,
                                        max_length = 300,
                                        top_p=0.95,
                                        temperature=0.95,
                                        num_return_sequences=1)
        final_out = tokenizer.decode(sample_output[0], skip_special_tokens=True)

    bot.send_message(chat_id=call.message.chat.id, text= final_out[len(prompt) + 2:])

# Запускаем постоянный опрос бота в Телеграме
bot.polling(none_stop=True, interval=0)

# Сайты, которые использовались для внедрения языкового модуля
# https://huggingface.co/shahp7575/gpt2-horoscopes
# https://github.com/shahp7575/gpt2-horoscopes/blob/master/generate_from_hub.py
# https://huggingface.co/Helsinki-NLP/opus-mt-ru-en
