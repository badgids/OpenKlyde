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

from aiohttp import ClientSession
from discord.ext import commands
from discord import app_commands
from discord import Interaction

# API Keys and Information
# Your API keys and tokens go here. Do not commit with these in place!
discord_api_key = "PUT_YOUR_DISCORD_API_KEY_HERE"

intents = discord.Intents.all()
intents.message_content = True

client = commands.Bot(command_prefix='$', intents=intents)

# Create our queues up here somewhere
queue_to_process_message = asyncio.Queue() # Process messages and send to LLM
queue_to_process_image = asyncio.Queue() # Process images from SD API
queue_to_send_message = asyncio.Queue() # Send messages to chat and the user


# Character Card (current character personality)
character_card = {}

# Global card for API information. Used with use_api_backend.
text_api = {}
image_api = {}

status_last_update = None
last_message_sent = datetime.datetime.now()

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

async def handle_image(message):
    # Mark the message as read (we know we're working on it)
    await message.add_reaction('✨')
    try:
        # Process each attachment (actually just one for now)
        for attachment in message.attachments:
            # Check if it is an image based on content type
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']):
                # Download the image bytes
                image_bytes = await attachment.read()  # Uses the read method from discord.Attachment class
                
                # Convert the image to base64
                base64_image = encode_image_to_base64(image_bytes)
                
                # Define the POST data
                post_data = {
                    'prompt': "USER: Describe what you see in this image: [img-1] \nASSISTANT: Here we have ",
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

async def bot_behavior(message):

    # If the bot wrote the message, don't do anything more!
    if message.author == client.user:
        return False
    
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
        'image': image_request
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
        data = llm_response['choices'][0]['text']
    
    llm_message = await functions.clean_llm_reply(data, content["user"], character_card["name"])
    
    queue_item = {"response": llm_message,"content": content}

    #if content["image"] == True:
    #    queue_to_process_image.put_nowait(queue_item)
        
    #else:
    queue_to_send_message.put_nowait(queue_item)

async def send_to_model_queue():
    global text_api
    
    while True:
        # Get the queue item that's next in the list
        content = await queue_to_process_message.get()
        
        # Add the message to the user's history
        await functions.add_to_conversation_history(content["user_input"], content["user"], content["user"])
        
        # Grab the data JSON we want to send it to the LLM
        await functions.write_to_log("Sending prompt from " + content["user"] + " to LLM model.")

        async with ClientSession() as session:
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

# Reply queue that's used to allow the bot to reply even while other stuff if is processing 
async def send_to_user_queue():
    while True:
    
        # Grab the reply that will be sent
        reply = await queue_to_send_message.get()
        
        # Add the message to user's history
        await functions.add_to_conversation_history(reply["response"], character_card["name"], reply["content"]["user"])
        
        # Update reactions
        await reply["content"]["message"].remove_reaction('✨', client.user)
        await reply["content"]["message"].add_reaction('✅')
        
        
        #if reply["content"]["image"]:
        #    image_file = discord.File(reply["image"])
        #    await reply["content"]["message"].channel.send(reply["response"], file=image_file, reference=reply["content"]["message"])
        #    os.remove(reply["image"])
        
        #else:
        await reply["content"]["message"].channel.send(reply["response"], reference=reply["content"]["message"])

        queue_to_send_message.task_done()

@client.event
async def on_ready():
    # Let owner known in the console that the bot is now running!
    print(f'Discord Bot is up and running.')
    
    global text_api
    global image_api
    global character_card
    
    text_api = await functions.set_api("text-default.json")
    print(text_api)
    image_api = await functions.set_api("image-default.json")
    
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
