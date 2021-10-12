__author__ = "Aurélien Bück-Kaeffer"
__version__ = "1.1"
__date__ = "12-10-2021"
# bot.py

import os
import discord
from neural_network import *
import json
import random

models = {}
jsons = {}

class AverageUser(discord.Client):

    async def on_ready(self):
        #When booting up
        print(f'{self.user} has connected to Discord!')

        for guild in self.guilds:
            print("Connected to : {}".format(guild))

            files = os.listdir("Models")
            if "{}.json".format(guild) not in files or "{}.hdf5".format(guild) not in files:
                #If no model has been created for a server yet
                print("No model found for {}".format(guild))
                continue
            
            #Loads every model and the data saved in jsons that comes with it
            models[guild.name] = load_model("Models/{}.hdf5".format(guild))
            with open("Models/{}.json".format(guild), 'r') as f:
                jsons[guild.name] = json.load(f)
            print("loaded model for {}".format(guild))

            
        print("finished loading")
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        
        
        message_content = message.content.split(" ")

        if message_content[0] == "|learn":
            #Command that creates a new network for the server and trains it on (number of channels in the server * argument) messages

            #TODO: Ajouter un caractere de fin de message

            #Goes through the n last messages in every channel in the server, n beng the argument specified in the command
            embed = discord.Embed(description = "Scraping channels...")
            await message.channel.send(embed=embed)

            data = ""

            for channel in message.guild.text_channels:
                async for msg in channel.history(limit = int(message_content[1])):
                    if msg.content != '':
                        data += "{}£\n".format(msg.content)
            
            embed = discord.Embed(description = "Preprocessing data...")
            await message.channel.send(embed=embed)

            #Makes the data usable by a neural network
            X, y, input_len, vocab_len, seq_length, n_patterns, num_to_char, char_to_num = preprocess_data(data)

            #Saves important values for generating text
            saving_json = {"num_to_chars" : num_to_char,
                            "char_to_nums" : char_to_num,
                            "message_proba" : 0.1,
                            "vocab_len" : vocab_len}
            with open('Models/{}.json'.format(message.guild), 'w') as f:
                json.dump(saving_json, f)

            embed = discord.Embed(description = "Training Model... This might be very long")
            await message.channel.send(embed=embed)

            #Trains the model
            model = create_model(X.shape[1], X.shape[2], y.shape[1])
            train(model, message.guild, X, y)

            #Adds the model to the list
            jsons[message.guild]=saving_json
            models[message.guild]=model

            embed = discord.Embed(description = "Finished training")
            await message.channel.send(embed=embed)
        
        elif message_content[0] == "|set_proba":

            files = os.listdir("Models")
            if "{}.json".format(message.guild) not in files or "{}.hdf5".format(message.guild) not in files:
                embed = discord.Embed(description = "You must first create a model")
                await message.channel.send(embed=embed)
                return

            jsons[message.guild.name]["message_proba"] = float(message_content[1])
            with open('Models/{}.json'.format(message.guild), 'w') as f:
                json.dump(jsons[message.guild.name], f)
            embed = discord.Embed(description = "Probability of sending a message changed to {}%".format(jsons[message.guild.name]["message_proba"]*100))
            await message.channel.send(embed=embed)
        
        elif message_content[0] == "|get_proba":
            files = os.listdir("Models")
            if "{}.json".format(message.guild) not in files or "{}.hdf5".format(message.guild) not in files:
                embed = discord.Embed(description = "You must first create a model")
                await message.channel.send(embed=embed)
                return

            embed = discord.Embed(description = "Probability of sending a message is {}%".format(jsons[message.guild.name]["message_proba"]*100))
            await message.channel.send(embed=embed)

        elif message_content[0] == "|help":
            title = "Average discord user manual"
            description = """This is a bot that uses natural language processing techniques to learn from the messages sent in a discord server and mimic the users, becoming the "average user"
                            
                            `|learn [number_of_messages]` : The bot will scrape the server, getting the [number_of_messages] last messages in EACH channel as training data.
                            This can, at most, give it [number_of_messages]*[number_of_channels] messages to train on.
                            Then, it preprocesses the data to make it usable as training material for the network. No message is saved during this process.
                            Finally, it creates a network assigned to the server and trains it on said preprocessed data. This might be extremely long if there are lots of messages (from hours to days).
                            This overwrites any previous model for that server, so proceed with caution if you already have something that works.
                            
                            `|set_proba [n]` : Takes a value n between 0 and 1 and sets the probability that the bot answers to any given message to than n value

                            `|get_proba` : Gives you the current probability of sending a message
                            
                            `|help` : Displays current message
                            
                            This bot was made by Aurélien Bück-Kaeffer"""
            embed = discord.Embed(title=title, description=description)
            await message.channel.send(embed=embed)

        else:
            #If a random message has been sent, there is a random chance that the bot will respond to it
            files = os.listdir("Models")
            if "{}.json".format(message.guild) not in files or "{}.hdf5".format(message.guild) not in files:
                return
            if jsons[message.guild.name]["message_proba"] > random.random():
                seed = ""
                async for msg in message.channel.history():
                    if msg.content != '':
                        seed += "{}£\n".format(msg.content)
                        seed = tokenize_words(seed)
                        if len(seed)>=100: break
                
                seed = seed[-100:]
                pattern = []
                for char in seed:
                    pattern.append(jsons[message.guild.name]["char_to_nums"][char])

                generated_text = ""

                for i in range(200):
                    x = numpy.reshape(pattern, (1, len(pattern), 1))
                    x = x / float(jsons[message.guild.name]["vocab_len"])
                    prediction = models[message.guild.name].predict(x, verbose=0)
                    index = numpy.argmax(prediction)
                    result = jsons[message.guild.name]["num_to_chars"][str(index)]

                    #sys.stdout.write(result)
                    generated_text += result

                    pattern.append(index)
                    pattern = pattern[1:len(pattern)]
                    if result == "A": break
                
                await message.channel.send(content=generated_text)


TOKEN = os.getenv('DISCORD_TOKEN')
client = AverageUser()

client.run(TOKEN)