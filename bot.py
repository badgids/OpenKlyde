import os
import discord
import requests
import json
import asyncio
import httpx
import random
import functions
import datetime
import base64
import io
import hashlib
import re
import unittest
import logging
from typing import List
from pydub import AudioSegment
from PIL import Image

# For xtts2 TTS
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from aiohttp import ClientSession
from aiohttp import ClientTimeout
from aiohttp import TCPConnector
from discord.ext import commands
from discord import app_commands
from discord import Interaction

# API Keys and Information
# Your API keys and tokens go here. Do not commit with these in place!
discord_api_key = "PUT_YOUR_API_KEY_HERE"

intents = discord.Intents.all()
intents.message_content = True

client = commands.Bot(command_prefix='$', intents=intents)

# Create our queues up here somewhere
queue_to_process_message = asyncio.Queue() # Process messages and send to LLM
queue_to_process_image = asyncio.Queue() # Process images from SD API
queue_to_send_message = asyncio.Queue() # Send messages to chat and the user

# Global TTS model variables
tts_config = None
tts_model = None
gpt_cond_latent = None
speaker_embedding = None

# Character Card (current character personality)
character_card = {}

# Global card for API information. Used with use_api_backend.
text_api = {}
image_api = {}

status_last_update = None
last_message_sent = datetime.datetime.now()

# Unit tests
class TestSplitDialogue(unittest.TestCase):
    
    def test_short_dialogue(self):
        self.assertEqual(split_dialogue("Hello", 175), ["Hello"])

    def test_exact_length_dialogue(self):
        dialogue = "a" * 175
        self.assertEqual(split_dialogue(dialogue, 175), [dialogue])

    def test_long_dialogue_without_punctuation(self):
        dialogue = "a" * 200
        self.assertEqual(split_dialogue(dialogue, 175), [dialogue])

    def test_long_dialogue_with_punctuation(self):
        dialogue = "a" * 173 + "." + "b" * 173 + "!"
        self.assertEqual(len(split_dialogue(dialogue, 175)), 2)

    def test_longer_dialogue_with_punctuation(self):
        dialogue = "a" * 170 + "." + "b" * 170 + "!"
        dialogue += "a" * 170 + "." + "b" * 170 + "!"
        dialogue += "a" * 170 + "." + "b" * 170 + "!"
        dialogue += "a" * 170 + "." + "b" * 170 + "!"
        dialogue += "a" * 170 + "." + "b" * 170 + "!"
        self.assertEqual(len(split_dialogue(dialogue, 175)), 10)

    def test_longer_dialogue_with_emoji(self):
        dialogue = "a" * 170 + "😊" + "b" * 170 + "!"
        dialogue += "a" * 170 + "." + "b" * 170 + "!"
        dialogue = "a" * 170 + "😊" + "b" * 170 + "!"
        dialogue += "a" * 170 + "." + "b" * 170 + "!"
        dialogue += "a" * 170 + "." + "b" * 170 + "!"
        self.assertEqual(len(split_dialogue(dialogue, 175)), 10)

    def test_dialogue_with_emoji(self):
        dialogue = "a" * 100 + "😊" + "b" * 74
        #expected_parts = ["a" * 100 + "😊", "b" * 74]
        self.assertEqual(len(split_dialogue(dialogue, 175)), 1)

async def update_status():

    global status_last_update
    now = datetime.datetime.now()
    
    # If status has never been updated, or it's been more than 30 seconds, update status
    if status_last_update == None or now - status_last_update > datetime.timedelta(seconds=30):
        
        data = await functions.check_bot_temps()
        activity = discord.Activity(type=discord.ActivityType.watching, name=data)
        await client.change_presence(activity=activity)
        status_last_update = datetime.datetime.now()

# Helper function to convert image to base64
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

async def convert_webp_bytes_to_png(image_bytes):
    with io.BytesIO(image_bytes) as image_file:
        with Image.open(image_file) as img:
            output_buffer = io.BytesIO()
            img.save(output_buffer, format="PNG")
            return output_buffer.getvalue()

async def handle_image(message):
    # Mark the message as read (we know we're working on it)
    await message.add_reaction('✨')
    try:
        # Process each attachment (actually just one for now)
        for attachment in message.attachments:
            # Check if it is an image based on content type
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']):
                # Download the image bytes
                image_bytes = await attachment.read()  # Uses the read method from discord.Attachment class

                # if .webp -> convert to PNG for llava
                if attachment.filename.lower().endswith('.webp'):
                    image_bytes = await convert_webp_bytes_to_png(image_bytes)
                
                # Convert the image to base64
                base64_image = encode_image_to_base64(image_bytes)
                
                # Define the POST data
                post_data = {
                    'prompt': "USER: Describe what you see and recognize in this image: [img-1] \nASSISTANT: ",
                    'n_predict': 256,
                    'image_data': [{"data": base64_image, "id": 1}],
                    'ignore_eos': False,
                    'temperature': 0.1
                }
                
                # Encode the data as JSON
                json_data = json.dumps(post_data)
                
                # Set the request headers
                headers = {
                    'Content-Type': 'application/json',
                }
                
                # Specify the URL
                llava_url = "http://localhost:8007/completion"
                
                # Perform the HTTP POST request for image analysis
                async with ClientSession() as session:
                    async with session.post(llava_url, headers=headers, data=json_data) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            image_description = response_data['content']
                            
                            # Send the response back to the Discord channel
                            #await message.channel.send(image_description, reference=message)
                            return image_description
                        else:
                            # Handle unexpected status code
                            errorstr = f"Error: The server responded with an unexpected status code: {response.status}"
                            await functions.write_to_log(errorstr)
                            return None

            
            else:
                # If no image is found
                await functions.write_to_log("No supported image attachments found.")
                return None
    
    except Exception as e:
        # Handle any other exception that was not explicitly caught
        error_msg = f"An error occurred: {str(e)}"
        await functions.write_to_log(error_msg)
        return None
    return None

# Dictionary to keep track of the bot's last message time and last mentioned channel by guild
bot_last_message_time = {}
bot_last_mentioned_channel = {}

async def bot_send_random_message_on_channel(channel):
    character = functions.get_character(character_card)
    
    global text_api

    #reply = await get_reply(message)
    #history = await functions.get_conversation_history(user, 15)
#    await functions.write_to_log("Assembling random msg prompt")
    prompt = await functions.create_prompt_for_random_message(character, character_card['name'], text_api)
#    await functions.write_to_log("Random message prompt assembled: " + prompt)
    
    queue_item = {
        'prompt': prompt,
        'message': None,
        'user_input': None,
        'user': None,
        'image': None,
        'channel': channel
    }
    
    queue_to_process_message.put_nowait(queue_item)

async def bot_behavior(message):
    global bot_last_message_time
    global bot_last_mentioned_channel

    # If the bot wrote the message, update last message time
    if message.author == client.user:
        if message.guild:
            bot_last_message_time[message.guild.id] = datetime.datetime.now()
        return False
    
    # If the bot is mentioned in a message and not a DM, update last mentioned channel
    if client.user.mentioned_in(message) and message.guild:
        bot_last_mentioned_channel[message.guild.id] = message.channel

# TODO: implement sending random messages to channel after timeout
    # If message not directed to bot
    if not client.user.mentioned_in(message):
        # Check if it's been 30 minutes since the bot last spoke
        # and the bot was mentioned in this guild
        if message.guild and message.guild.id in bot_last_message_time:
            time_since_bot_last_spoke = datetime.datetime.now() - bot_last_message_time[message.guild.id]
            if time_since_bot_last_spoke.total_seconds() >= 200000:  # 2 minutes * 60 seconds
                # Check if there's a recorded channel from a bot mention
                if message.guild.id in bot_last_mentioned_channel:
#                    await functions.write_to_log("Initiate random message send")
                    await bot_send_random_message_on_channel(bot_last_mentioned_channel[message.guild.id])
#                    await functions.write_to_log("returned from bot_send_random_message_on_channel")
                    # Update the last message time after sending a pun
                    bot_last_message_time[message.guild.id] = datetime.datetime.now()
                    return True



    # If the bot is mentioned in a message and message has attachment, reply to the message parsing the attachment
    if client.user.mentioned_in(message) and message.attachments:
        img_description = await handle_image(message)  # Function to handle image processing
        if img_description:
            await bot_answer(message, img_description)
            # Set the time of last sent message to right now
            last_message_sent = datetime.datetime.now()
            return True
        else:
            return False

    # If the bot is mentioned in a message, reply to the message
    if client.user.mentioned_in(message):
        await bot_answer(message)
        
        # Set the time of last sent message to right now
        last_message_sent = datetime.datetime.now()
        return True
    
    #If someone DMs the bot and message has attachment, reply to them in the same DM parsing the attachment
    if message.guild is None and not message.author.bot and message.attachments:
        img_description = await handle_image(message)  # Function to handle image processing
        if img_description:
            await bot_answer(message, img_description)
            return True
        else:
            return False

    #If someone DMs the bot, reply to them in the same DM
    if message.guild is None and not message.author.bot:
        await bot_answer(message)
        return True
        
    # If I haven't spoken for 30 minutes, say something in the last channel where I was pinged (not DMs) with a pun or generated image
    # If someone speaks in a channel, there will be a three percent chance of answering (only in chatbots and furbies)
    # If I'm bored, ping someone with a message history
    # If I have a reminder, pop off the reminder in DMs at selected time and date
    # If someone asks me about the weather, look up weather at a given zip code/location
    # If someone asks me about a wikipedia article, provide the first 300 words from the article's page
    # Google wikipedia and add info to context before answering
    # If someone asks for a random number, roll some dice
    # If someone wants me to be chatty, change personality on the fly to chatterbox
    # If someone asks for a meme, generate an image of a meme on the fly
    # If playing a game or telling a story, add an image to the story
    
    return False

async def bot_answer(message, image_description=None):
    # Mark the message as read (we know we're working on it)
    await message.add_reaction('✨')
    
    user = message.author.display_name
    user= user.replace(" ", "")
    
    # Clean the user's message to make it easy to read
    user_input = functions.clean_user_message(message.clean_content)
    
    #Is this an image request?
    #image_request = functions.check_for_image_request(user_input)
    character = functions.get_character(character_card)
    image_request = False
    
    global text_api

    #if image_request:
    #    prompt = await functions.create_image_prompt(user_input, character, text_api)
    #else:
    reply = await get_reply(message)
    history = await functions.get_conversation_history(user, 15)
    prompt = await functions.create_text_prompt(user_input, user, character, character_card['name'], history, reply, text_api, image_description)
        
    
    queue_item = {
        'prompt': prompt,
        'message': message,
        'user_input': user_input,
        'user': user,
        'image': image_request,
        'channel': None
    }
    
    queue_to_process_message.put_nowait(queue_item)

# Get the reply to a message if it's relevant to the conversation
async def get_reply(message):
    reply = ""

    # If the message reference is not none, meaning someone is replying to a message
    if message.reference is not None:
        # Grab the message that's being replied to
        referenced_message = await message.channel.fetch_message(message.reference.message_id)

        #Verify that the author of the message is bot and that it has a reply
        if referenced_message.reference is not None and referenced_message.author == client.user: 
        # Grab that other reply as well
            referenced_user_message = await message.channel.fetch_message(referenced_message.reference.message_id)

            # If the author of the reply is not the same person as the initial user, we need this data
            if referenced_user_message.author != message.author:
                reply = referenced_user_message.author.display_name + ": " + referenced_user_message.clean_content + "\n"
                reply = reply + referenced_message.author.display_name + ": " + referenced_message.clean_content + "\n"
                reply = functions.clean_user_message(reply)

                return reply
        
        #If the referenced message isn't from the bot, use it in the reply
        if referenced_message.author != client.user: 
            reply = referenced_message.author.display_name + ": " + referenced_message.clean_content + "\n"

            return reply

    return reply

async def handle_llm_response(content, response):
    
    llm_response = json.loads(response)
    
    try:
        data = llm_response['results'][0]['text']
    except KeyError:
        data = llm_response['choices'][0]['message']['content']
    
    llm_message = data
    # do clean only for replies
    if not content['channel']:
        llm_message = await functions.clean_llm_reply(data, content["user"], character_card["name"])
    else:
        llm_message = llm_message.replace(character_card["name"] + ":","")
        llm_message = llm_message.strip()
#        await functions.write_to_log("Random message cleaned: " + llm_message)
    
    queue_item = {"response": llm_message,"content": content}
    if not llm_message:
        await functions.write_to_log("hm, llm_message is empty..")
        return

    #if content["image"] == True:
    #    queue_to_process_image.put_nowait(queue_item)
        
    #else:
    queue_to_send_message.put_nowait(queue_item)

async def send_to_model_queue():
    global text_api
    
    while True:
        # Get the queue item that's next in the list
        content = await queue_to_process_message.get()
        await functions.write_to_log("send_to_model_queue()")
        
        # Add the message to the user's history (if this is a reply)
        if not content['channel']:
            await functions.add_to_conversation_history(content["user_input"], content["user"], content["user"])
        
        # Grab the data JSON we want to send it to the LLM
        if not content['channel']:
            await functions.write_to_log("Sending prompt from " + content["user"] + " to LLM model.")
        else:
            await functions.write_to_log("Sending prompt for " + content['channel'].name + " to LLM model.")

        timeout = ClientTimeout(total=600)
        connector = TCPConnector(limit_per_host=10)
        async with ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(text_api["address"] + text_api["generation"], headers=text_api["headers"], data=content["prompt"]) as response:
                response = await response.read()
                
                # Do something useful with the response
                await handle_llm_response(content, response)

                queue_to_process_message.task_done()

async def send_to_stable_diffusion_queue():
    global image_api

    while True:
    
        image_prompt = await queue_to_process_image.get()
        
        data = image_api["parameters"]
        data["prompt"] += image_prompt["response"]
        data_json = json.dumps(data)

        await functions.write_to_log("Sending prompt from " + image_prompt["content"]["user"] + " to Stable Diffusion model.")
        
        async with ClientSession() as session:
            async with session.post(image_api["link"], headers=image_api["headers"], data=data_json) as response:
                response = await response.read()
                sd_response = json.loads(response)
                
                image = functions.image_from_string(sd_response["images"][0])
                
                queue_item = {
                    "response": image_prompt["response"],
                    "image": image,
                    "content": image_prompt["content"]
                }
                queue_to_send_message.put_nowait(queue_item)
                queue_to_process_image.task_done()

# New function to handle TTS generation
async def generate_tts(text):
    await functions.write_to_log("Generating TTS for the given text...")
    out = tts_model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.70,
    )
    md5hash = hashlib.md5(text.encode('utf-8'))
    md5hash_hex = md5hash.hexdigest()
    audio_path = "tts_output_" + md5hash_hex + ".wav"
    torchaudio.save(audio_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    return audio_path

def strip_emoji(part: str):
    pattern_new = (
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]{1,}"
    )
    return re.sub(pattern_new, '', part)

# TODO try a different approach for chunking
# Maybe do the inverse - treat any alphanum character that is not ?!.<emoji> as not-a-sentence-ending
# That way if a char is detected that is not alphanum, or is ?!.<emoji> treat it or a group of it, as sentence ending

# Another way is to first chunk into sentences - 1 sentence in element
# Then do another pass and try to group them in chunks less than 175
# So only forward passes

# find biggest chunk (that is at most max_length) counting from the end of given string
def get_start_idx_of_biggest_chunk_from_end(currStr: str, max_length):
    # match any unicode emoji, or !, ?, or one or more . together
    #pattern = r'[\U0001F100-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U00002600-\U000027BF\U00002b50\U00002b55\U00002328\U0000232a\U0001f601-\U0001f64f\U00002702-\U000027b0\U00002600-\U000027BF\U0001f300-\U0001f64F\U0001f680-\U0001f6c5]|\!|\?|\.{1,}'
    pattern_new = (
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]|\!|\?|\.{1,}"
    )
    #pattern = r'[-\U0001f6c5]|\!|\?|\.{1,}'

    # if whole string not more than 175 - return it as a chunk
    if len(currStr) <= max_length:
        #print("return whole")
        return 0

    # find start idx of biggest chunk from end
    current_chunk_len = 0
    for i in range(len(currStr) - 1, -1, -1):
        if i == 0:
            #functions.write_to_log("get_start_idx_of_biggest_chunk_from_end() reached beginning - returning it whole")
            # should not happen, reached the beginning, return whole string
            return 0
        if current_chunk_len > max_length:
            # move idx forward to a sentence boundary where chunk would be <175
            separator_in_progress = False
            for j in range(i, len(currStr)):
                #print("backtracking,currstr[i]=" + currStr[j] + ", j=" + str(j) + " i=" + str(i))
                if re.match(pattern_new, currStr[j]):
                    separator_in_progress = True # could be a multi-char separator
                else:
                    if separator_in_progress == True:
                        #print("ended backtracking, returning " + str(j))
                        separator_in_progress = False
                        return j

                # if somehow chunk was longer than limit without any sentence endings, oh well, return the whole thing
                if j == (len(currStr) - 1):
                    functions.write_to_log("sentence longer than maxlength - returning it whole")
                    return 0
                    
        current_chunk_len += 1

    return 0

# This function returns a list of strings, each string being a chunk of the original dialogue that adheres to the requirements (chunk <= 175).
# Please note that this function assumes that the dialogue does not contain any sentences longer than the max length without any separators. If such sentences are possible, the function will need to be modified to handle them.
# This function should work well for most human dialogue and written chat logs as long as they follow usual punctuation and emoji conventions. 
# However, it may not be perfect. For example, it may not handle correctly dialogue that contains abbreviations with periods (e.g., "Mr.", "Mrs.") or numbers with periods (e.g., "1.5", "100.00").
# If such cases are common in your data, you will need to write additional code to handle them.
def split_dialogue(dialogue: str, max_length) -> List[str]:

    if len(dialogue) <= max_length:
        return [dialogue]

    dialogue_parts = []
    start_idx = len(dialogue)
    while True:
        currStr = dialogue[:start_idx]
        #print("start_idx = " + str(start_idx))
        #print("currStr = " + currStr)
        start_idx = get_start_idx_of_biggest_chunk_from_end(currStr, max_length)
        dialogue_parts.append(currStr[start_idx:])
        if start_idx == 0:
            break

    dialogue_parts.reverse()
    return dialogue_parts
        

# Reply queue that's used to allow the bot to reply even while other stuff if is processing 
async def send_to_user_queue():
    while True:
    
        # Grab the reply that will be sent
        reply = await queue_to_send_message.get()
        
        if not reply["content"]["channel"]:
            # Add the message to user's history
            await functions.add_to_conversation_history(reply["response"], character_card["name"], reply["content"]["user"])

            # Update reactions
            await reply["content"]["message"].remove_reaction('✨', client.user)
            await reply["content"]["message"].add_reaction('✅')

        # After getting the dialogue, split it
#        await functions.write_to_log("reply response is len " + str(len(reply["response"])))
        dialogue_parts = split_dialogue(reply["response"], 200)
#        await functions.write_to_log("dialogue parts is len " + str(len(dialogue_parts)))

        # Generate TTS audio for each part
        audio_parts = []
        for part in dialogue_parts:
            # better not send emojis to TTS
            part = strip_emoji(part)
#            await functions.write_to_log("part = " + part)
            audio_path = await generate_tts(part)
            audio_parts.append(AudioSegment.from_wav(audio_path))
            os.remove(audio_path)

        #print("finished sending tts")

        # Create a silent audio segment of 0.5 seconds (500 milliseconds)
        silence = AudioSegment.silent(duration=800)     

        # Add the silent segment between each pair of audio segments
        combined_audio = sum(x for y in zip(audio_parts, [silence]*len(audio_parts)) for x in y)

        # Save the combined audio file
        md5hash = hashlib.md5(reply["response"].encode('utf-8'))
        md5hash_hex = md5hash.hexdigest()
        combined_audio_path = "tts_output_" + md5hash_hex + ".wav"
        combined_audio.export(combined_audio_path, format="wav")

        # Send the combined audio file
        audio_file = discord.File(combined_audio_path)

       # # Generate TTS audio from the response
       # audio_path = await generate_tts(reply["response"])
       # audio_file = discord.File(audio_path)
        
        #if reply["content"]["image"]:
        #    image_file = discord.File(reply["image"])
        #    await reply["content"]["message"].channel.send(reply["response"], file=image_file, reference=reply["content"]["message"])
        #    os.remove(reply["image"])
        
        #else:

        if not reply["content"]["channel"]:
            await reply["content"]["message"].channel.send(reply["response"], file=audio_file, reference=reply["content"]["message"])
            #await reply["content"]["message"].channel.send(reply["response"], reference=reply["content"]["message"])
        else:
            # send random message on channel
            #await functions.write_to_log("Sending random message.")
            print("Sending random message.")
            await reply["content"]["channel"].send(reply["response"])


        queue_to_send_message.task_done()

@client.event
async def on_ready():
    # Let owner known in the console that the bot is now running!
    print(f'Discord Bot is up and running.')
    
    global text_api
    global image_api
    global character_card
    global tts_config, tts_model, gpt_cond_latent, speaker_embedding

    logging.basicConfig(level=logging.DEBUG)

    # Load TTS model
    await functions.write_to_log("Loading TTS model...")
    tts_config = XttsConfig()
    tts_config.load_json("xtts/config.json")
    tts_model = Xtts.init_from_config(tts_config)
    tts_model.load_checkpoint(tts_config, checkpoint_dir="./xtts/model", use_deepspeed=False)
    tts_model.cuda() 

    # Compute speaker latents
    await functions.write_to_log("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[r"xtts/scarlett24000.wav"])
    
    text_api = await functions.set_api("text-default.json")
    image_api = await functions.set_api("image-default.json")
    
    if text_api["name"] != "openai":
        api_check = await functions.api_status_check(text_api["address"] + text_api["model"], headers=text_api["headers"])

    character_card = await functions.get_character_card("default.json")
    
    #AsynchIO Tasks
    asyncio.create_task(send_to_model_queue())
    asyncio.create_task(send_to_stable_diffusion_queue())
    asyncio.create_task(send_to_user_queue())
    
    # Sync current slash commands (commented out unless we have new commands)
    client.tree.add_command(personality)
    client.tree.add_command(history)
    client.tree.add_command(character)
    client.tree.add_command(parameters)
    await client.tree.sync()
        
    # Check bot temps and update bot status accordingly
    await update_status()

@client.event
async def on_message(message):
    
    if message is None:
        return
    # Update hardware status
    await update_status()
    
    # Bot will now either do or not do something!
    await bot_behavior(message)
        
# Slash command to update the bot's personality
personality = app_commands.Group(name="personality", description="View or change the bot's personality.")

@personality.command(name="view", description="View the bot's personality profile.")
async def view_personality(interaction):
    # Display current personality.
    await interaction.response.send_message("The bot's current personality: **" + character_card["persona"] + "**.")
    
@personality.command(name="set", description="Change the bot's personality.")
@app_commands.describe(persona="Describe the bot's new personality.")
async def edit_personality(interaction, persona: str):
    global character_card
            
    # Update the global variable
    old_personality = character_card["persona"]
    character_card["persona"] = persona
        
    # Display new personality, so we know where we're at
    await interaction.response.send_message("Bot's personality has been updated from \"" + old_personality + "\" to \"" + character_card["persona"] + "\".")

@personality.command(name="reset", description="Reset the bot's personality to the default.")
async def reset_personality(interaction):
    global character_card
            
    # Update the global variable
    old_personality = character_card["persona"]
    character_card = await functions.get_character_card("default.json")
        
    # Display new personality, so we know where we're at
    await interaction.response.send_message("Bot's personality has been updated from \"" + old_personality + "\" to \"" + character_card["persona"] + "\".")

# Slash commands to update the conversation history    
history = app_commands.Group(name="conversation-history", description="View or change the bot's personality.")

@history.command(name="reset", description="Reset your conversation history with the bot.")
async def reset_history(interaction):
    user = str(interaction.user.display_name)
    user= user.replace(" ", "")

    file_name = functions.get_file_name("context", user + ".txt")

    # Attempt to remove the file and let the user know what happened.
    try:
        os.remove(file_name)
        await interaction.response.send_message("Your conversation history was deleted.")
    except FileNotFoundError:
        await interaction.response.send_message("There was no history to delete.")
    except PermissionError:
        await interaction.response.send_message("The bot doesn't have permission to reset your history. Let bot owner know.")
    except Exception as e:
        print(e)
        await interaction.response.send_message("Something has gone wrong. Let bot owner know.")

@history.command(name="view", description=" View the last 20 lines of your conversation history.")
async def view_history(interaction):
    # Get the user who started the interaction and find their file.

    user = str(interaction.user.display_name)
    user= user.replace(" ", "")

    file_name = functions.get_file_name("context", user + ".txt")
    
    try:
        with open(file_name, "r", encoding="utf-8") as file:  # Open the file in read mode
            contents = file.readlines()
            contents = contents[-20:]
            history_string = ''.join(contents)
            await interaction.response.send_message(history_string)
    except FileNotFoundError:
        await interaction.response.send_message("You have no history to display.")
    except Exception as e:
        print(e)
        await interaction.response.send_message("Message history is more than 2000 characters and can't be displayed.")

# Slash commands for character card presets (if not interested in manually updating) 
character = app_commands.Group(name="character-cards", description="View or changs the bot's current character card, including name and image.")

# Command to view a list of available characters.
@character.command(name="change", description="View a list of current character presets.")
async def change_character(interaction):
    
    # Get a list of available character cards
    character_cards = functions.get_file_list("characters")
    options = []
    
    # Verify the list is not currently empty
    if not character_cards:
        await interaction.response.send_message("No character cards are currently available.")
        return
        
    # Create the selector list with all the available options.
    for card in character_cards:
        options.append(discord.SelectOption(label=card, value=card))

    select = discord.ui.Select(placeholder="Select a character card.", options=options)
    select.callback = character_select_callback
    view = discord.ui.View()
    view.add_item(select)

    # Show the dropdown menu to the user
    await interaction.response.send_message('Select a character card', view=view, ephemeral=True)

async def character_select_callback(interaction):
    
    await interaction.response.defer()
    
    # Get the value selected by the user via the dropdown.
    selection = interaction.data.get("values", [])[0]
        
    # Adjust the character card for the bot to match what the user selected.
    global character_card
    
    character_card = await functions.get_character_card(selection)
    
    # Change bot's nickname without changing its name
    guild = interaction.guild
    me = guild.me
    await me.edit(nick=character_card["name"])
        
    # Let the user know that their request has been completed
    await interaction.followup.send(interaction.user.name + " updated the bot's personality to " + character_card["persona"] + ".")

# Slash commands for character card presets (if not interested in manually updating) 
parameters = app_commands.Group(name="model-parameters", description="View or changs the bot's current LLM generation parameters.")

# Command to view a list of available characters.
@parameters.command(name="change", description="View a list of available generation parameters.")
async def change_parameters(interaction):
    
    # Get a list of available character cards
    presets = functions.get_file_list("configurations")
    options = []
    
    # Verify the list is not currently empty
    if not presets:
        await interaction.response.send_message("No configurations are currently available. Please contact the bot owner.")
        return
        
    # Create the selector list with all the available options.
    for preset in presets:
        if preset.startswith("text"):
            options.append(discord.SelectOption(label=card, value=card))

    select = discord.ui.Select(placeholder="Select a character card.", options=options)
    select.callback = parameter_select_callback
    view = discord.ui.View()
    view.add_item(select)

    # Show the dropdown menu to the user
    await interaction.response.send_message('Select a character card', view=view, ephemeral=True)

async def parameter_select_callback(interaction):
    
    await interaction.response.defer()
    
    # Get the value selected by the user via the dropdown.
    selection = interaction.data.get("values", [])[0]
    
    # Adjust the character card for the bot to match what the user selected.
    global text_api
    text_api = await functions.set_api(selection)
    api_check = await functions.api_status_check(text_api["address"] + text_api["model"], headers=text_api["headers"])
    
    # Let the user know that their request has been completed
    await interaction.followup.send(interaction.user.name + " updated the bot's sampler parameters. " + api_check)


client.run(discord_api_key)
#unittest.main()
