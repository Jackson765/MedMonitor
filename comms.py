import asyncio
import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import time
import sys
import time

# Load environment variables from .env file
load_dotenv()

# Access the variables
botKey = os.getenv("KEY")
channelID = os.getenv("CHANNEL")

cooldown = False
dangerTime = 0
# 1. Setup Intents
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)
dangerValue = 0

async def inDanger():
    print(f'Logged in as {bot.user}')
    
    channel = bot.get_channel(int(channelID))
    
    if channel:
        file = discord.File("frame.jpg") 

        await channel.send("Someone is in danger! ⚠️😬💀", file=file)

@bot.event
async def on_ready():
    global dangerValue, cooldown, dangerTime
    print("bot ready")
    while (True):
        if (cooldown and dangerTime <= time.time()):
            cooldown == False
        if (dangerValue == 10 and not cooldown):
            print("Danger recieved")
            dangerValue = 0
            dangerTime = time.time() + 300000
            cooldown = True
            await inDanger()
        await asyncio.sleep(1) 

def setDanger():
    global dangerValue
    dangerValue += 1

def setDangerToZero():
    global dangerValue
    dangerValue = 0

def runBot():
    bot.run(botKey)