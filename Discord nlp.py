import discord
from discord.ext import commands
from transformers import pipeline
import re


toxicity_classifier = pipeline('text-classification', model='unitary/toxic-bert')


def preprocess_text(text):
 
    text = text.lower()
    
    replacements = {
        '@': 'a', '$': 's', '!': 'i', '1': 'i', '0': 'o', '3': 'e', '5': 's'
    }
    
    for special, normal in replacements.items():
        text = text.replace(special, normal)

    return text

def is_toxic(text):
    cleaned_text = preprocess_text(text)  
    result = toxicity_classifier(cleaned_text)
    
    print(f"Processed Text: {cleaned_text} | Model Output: {result}")
    
    return result[0]['label'] == 'toxic' and result[0]['score'] > 0.9 
intents = discord.Intents.default()
intents.message_content = True  

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Bot is online as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  

    if is_toxic(message.content):
        try:
            await message.delete()
            await message.channel.send(f"{message.author.mention}, your message was detected as toxic and was deleted. Please follow the community guidelines.")
        except discord.errors.Forbidden:
            print("⚠️ Bot lacks permission to delete messages.")

    await bot.process_commands(message) 

bot.run()
