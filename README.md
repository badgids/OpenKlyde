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

----NEW----: Elbios changes

Rough notes - I will polish the README later:

### XTTS
- Current revision uses XTTS2 (uses the TTS Python library, lookup on coqui.ai)
    - if you do not want to use TTS pass --no-tts like so: 'python bot.py --no-tts'
- XTTS2 uses some RAM/VRAM so bear that in mind
- for setup I used windows gpu steps from https://github.com/daswer123/xtts-api-server (no need to clone xtts-api-server, just do the install steps from its README)
- definitely need 'pip install pillow' and 'pip install TTS'
- ffmpeg must be in PATH environmental variable (download if you dont have it)
- create 'xtts' folder and put a short wave file with your voice sample as 'scarlett24000.wav'
- you can use any clean voice sample, 5-15seconds is best
- add files from here: https://huggingface.co/coqui/XTTS-v2/tree/main
- this should be the xtts dir contents:
```
$ find xtts
xtts
xtts/config.json
xtts/model
xtts/model/dvae.pth
xtts/model/mel_stats.pth
xtts/model/model.pth
xtts/model/vocab.json
xtts/scarlett24000.wav
```
- CUDA toolkit might be required, not sure

### LLAVA image recognition
- The bot supports Llava image recognition - you can send the bot an image and it will describe it and refer to it.
- if you dont use this feature, no need to have llava running
- otherwise, get llamacpp portable binaries and have this running in a terminal:
```
./server.exe -c 2048 -ngl 43 -nommq -m ./models/ggml-model-q4_k.gguf --host 0.0.0.0 --port 8007 --mmproj ./models/mmproj-model-f16.gguf
```
- the ggml models files you can get from Llava repo, also ShareGPT would also work

### Stable Diffusion
- bot supports stable diffusion but I recommend using SDXL as the bot will send a prompt to SD in natural language and previous SD (like SD1.5) would struggle

### Koboldcpp, OpenAI API, Mistral API
- The bot supports koboldcpp API but also OpenAI-compatible backends. Look in configurations folder for examples. Put your API key in the 'Bearer' line

### Additional configuration needed to run the bot
- Discord API key in bot.py in the global variable
- If using openai/mistral - API key in Bearer line in configuration/text-default.json
- fill in characters/default.json with your character prompt
- optionally in functions.py in function get_character() you can fill out the 'examples' array with examples of dialogue you want the bot to follow


## Instructions:

To run this bot:

1. Load the LLM model of your choice in Koboldcpp
2. Download this repository [OpenKlyde](https://github.com/badgids/OpenKlyde)
3. Open bot.py and at replace API_KEY with your bot's API key
4. Install the requirements. I suggest using an Anaconda or Miniconda instance.
    ```pip install -r requirements.txt```
5. Run the bot with `python bot.py`
    - optionally with --no-tts flag

Cheers!

## ToDo:

- [ ] Make a better README.
- [ ] Make switching between Koboldcpp or Oobabooga Textgen-ui more mainstream.
- [ ] Enable support for Character.ai, TavernAI, SillyTavern, etc. character formats.
- [ ] Add more standard Discord Bot features. (music, games, moderation, etc.)
