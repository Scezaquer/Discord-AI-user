# The Average Discord User
This is the code for a discord bot that uses the messages sent on a discord server to train a natural language processing AI, which will then send messages as well imitating what it has seen, becoming the "average discord user" of that server.

This project uses the discord, keras, nltk and numpy libraries.

To run this you need to add a .env file containing the following text in the main directory:
 ```
# .env
DISCORD_TOKEN=your_bot's_token_here
```

# Commands
This bot has 4 commands:

`|learn [number_of_messages]` : The bot will scrape the server, getting the [number_of_messages] last messages in EACH channel as training data.
                            This can, at most, give it [number_of_messages]*[number_of_channels] messages to train on.
                            Then, it preprocesses the data to make it usable as training material for the network. No message is saved during this process.
                            Finally, it creates a network assigned to the server and trains it on said preprocessed data. This might be extremely long if there are lots of messages (from hours to days).
                            This overwrites any previous model for that server, so proceed with caution if you already have something that works.
                            I do not recommend putting a number above 1000, since it can take days to train the model depending on the number of channels in the server.
                            
`|set_proba [n]` : Takes a value n between 0 and 1 and sets the probability that the bot answers to any given message to than n value

`|get_proba` : Gives you the current probability of sending a message
                            
`|help` : Displays these informations about the commands

# Files
There are two source files in this project:

bot.py, which takes care of all the interfacing with the discord API. This is the file to start to launch the bot.

neural_network.py, which contains all the functions related to the actual neural-network, like tokenizing (and preprocessing in general), creating, training the model

When training, the neural network performs 10 epochs. If interrupted and at least 1 epoch has been completed, the best version of the model will be saved and usable upon restart of the bot.


# How to use

-Create a discord bot

-Add a .env file containing information specified higher in this document

-install all dependencies (discord, keras, nltk and numpy libraries)

-Add the bot to one of your servers (it requires admin privileges)

-Run the bot.py file

-Type the `|learn x` command in your discord server (replace x by the number of messages to be collected in each channel, I don't recommend more than 1000)

-Wait for the agent to train (this might be extremely long, you can see the progress on your terminal). If you shutdown the bot during this phase, the training will be interrupted and progress might be lost.

-Once the training is finished, use the `|get_proba` and `|set_proba` commands to tune the probability of the bot responding to a message when it is sent

Once a message is sent, the bot has that probability of sending a message back, except if it crashes because of an unknown character for example. I did not make any failsafes.

I do not host this bot myself because I am not google. I can't afford to have people train neural networks on my computer. I can barely train them for my own use.
