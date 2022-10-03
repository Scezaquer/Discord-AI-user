__author__ = "Aurélien Bück-Kaeffer"
__version__ = "1.1"
__date__ = "12-10-2021"
# bot.py

import os
import discord
import random
from dotenv import load_dotenv
import cohere

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
COHERE_TOKEN = os.getenv('COHERE_TOKEN')

co = cohere.Client(COHERE_TOKEN)

text_message_settings = {"model":'xlarge',
        "prompt":"",
        "max_tokens":100,
        "temperature":0.7,
        "k":0,
        "p":0.75,
        "frequency_penalty":0,
        "presence_penalty":0,
        "stop_sequences":["--"],
        "return_likelihoods":'NONE'}

class AverageUser(discord.Client):
    async def on_ready(self):
        #When booting up
        print(f'{self.user} has connected to Discord!')

        for guild in self.guilds:
            print("Connected to : {}".format(guild))
            
        print("finished loading")


    async def on_message(self, message):
        #When receiving a message

        #TODO: Day/Night cycle, varying presence depending on time of day.
        #TODO: Ability to hold PM conversation
        if message.author == self.user:
            return

        message_content = message.content

        tmp = []

        async for msg in message.channel.history(limit = 20):
            if msg.content != '':
                tmp.append(f"{msg.author.name}:{msg.content}")
        
        data=tmp[0]
        for x in range(len(tmp)-1):
            data = tmp[x+1]+ "\n" + tmp[x] + "\n--\n" + data

        data += f"\n{self.user.name}:"
        text_message_settings["prompt"]=data
        settings = text_message_settings

        result = co.generate(
        model=settings["model"],
        prompt=settings["prompt"],
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        k=settings["k"],
        p=settings["p"],
        frequency_penalty=settings["frequency_penalty"],
        presence_penalty=settings["presence_penalty"],
        stop_sequences=settings["stop_sequences"],
        return_likelihoods=settings["return_likelihoods"]).generations[0].text[:-3]

        await message.channel.send(content=result)

client = AverageUser()

client.run(TOKEN)