__author__ = "Aurélien Bück-Kaeffer"
__version__ = "0.1"
__date__ = "10-10-2021"
# bot.py

import os
import discord
import pandas as pd
from neural_network import *
import json

models = {}
jsons = {}

class AverageUser(discord.Client):
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        for guild in self.guilds:
            print("Connected to : {}".format(guild))
            try:
                models[guild] = load_model("Models/{}.hdf5".format(guild))
                with open("Models/{}.json".format(guild), 'r') as f:
                    jsons[guild] = json.load(f)
                print("loaded model for {}".format(guild))
            except:
                print("No model found for {}".format(guild))
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        
        message_content = message.content.split(" ")

        if message_content[0] == "|learn":

            embed = discord.Embed(description = "Scraping channels...")
            await message.channel.send(embed=embed)

            data = ""

            for channel in message.guild.text_channels:
                async for msg in channel.history(limit = int(message_content[1])):
                    if msg.content != '':
                        data += "{}:\n{}\n\n".format(msg.author.name, msg.content)
            
            embed = discord.Embed(description = "Preprocessing data...")
            await message.channel.send(embed=embed)

            X, y, input_len, vocab_len, seq_length, n_patterns, num_to_char, char_to_num = preprocess_data(data)

            saving_json = {"num_to_chars" : num_to_char,
                            "char_to_nums" : char_to_num}
            with open('Models/{}.json'.format(message.guild), 'w') as f:
                json.dump(saving_json, f)

            embed = discord.Embed(description = "Training Model... This might be very long")
            await message.channel.send(embed=embed)

            model = create_model(X.shape[1], X.shape[2], y.shape[1])
            train(model, message.guild, X, y)
        
        else:
            pass


TOKEN = os.getenv('DISCORD_TOKEN')
client = AverageUser()

client.run(TOKEN)