# OpenKlyde - A Self Hosted AI Discord Bot

OpenKyde is an AI Discord Bot!

It incorporates an AI Large Language Model (LLM) into a discord bot by making API calls to a Koboldcpp instance. It can also work with Oobabooga.

You will need an instance of Koboldcpp running on your machine. In theory, you should also be able to connect it to the Horde,
but I haven't tested the implementation yet.

As of now this bot is only a chat bot, but it can also generate images with Automatic1111 Stable Diffusion using the following keywords after mentioning the bot:
"send|create|give|generate|draw|snap|show|take|message" and "image|picture|photo|photogragh|pic|drawing|painting|screenshot"

## Prerequisites

Download Koboldcpp here:
[Koboldcpp](https://github.com/LostRuins/koboldcpp)

If you want to generate images, you'll also need Automatic1111 Stable Diffusion runing with --listen and --api optons enabled on launch.
Download Automatic1111 here:
[Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)


## Instructions:

To run this bot:

1. Load the LLM model of your choice in Koboldcpp
2. Download this repository [OpenKlyde](https://github.com/badgids/OpenKlyde)
3. Open bot.py and at replace API_KEY with your bot's API key
4. Install the requirements. I suggest using an Anaconda or Miniconda instance.
    ```pip install -r requirements.txt```
5. Run the bot with `python bot.py`

Cheers!

## ToDo:

- [ ] Make a better README.
- [ ] Make switching between Koboldcpp or Oobabooga Textgen-ui more mainstream.
- [ ] Enable support for Character.ai, TavernAI, SillyTavern, etc. character formats.
- [ ] Add more standard Discord Bot features. (music, games, moderation, etc.)
