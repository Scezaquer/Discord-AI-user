__author__ = "Aurélien Bück-Kaeffer"
__version__ = "0.1"
__date__ = "10-10-2021"
# bot.py
import os
import discord
import pandas as pd

class AverageUser(discord.Client):
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        for guild in self.guilds:
            print("Connected to : {}".format(guild))
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        
        message_content = message.content.split(" ")

        if message_content[0] == "|learn":

            embed = discord.Embed(description = "Scraping channels...")
            await message.channel.send(embed=embed)

            data = pd.DataFrame(columns=['content', 'author'], dtype=object)

            for channel in message.guild.text_channels:
                async for msg in channel.history(limit = int(message_content[1])):
                    if msg.content != '':
                        data = data.append({'content': msg.content,
                                    'author': msg.author.name}, ignore_index=True)
            
            embed = discord.Embed(description = "Saving data...")
            await message.channel.send(embed=embed)

            data.to_csv("Message Database/database.csv", encoding='utf-16')
            
            embed = discord.Embed(description = "Data Saved")
            await message.channel.send(embed=embed)


TOKEN = os.getenv('DISCORD_TOKEN')
client = AverageUser()

client.run(TOKEN)