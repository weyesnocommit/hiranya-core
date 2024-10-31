# insalltka

https://drive.google.com/file/d/1DCNqGsRRIibbbp9UZIDGqsqe1vaa_1x0/view?usp=sharing

Nake u download weights and put d zlib file in weights folder o

# About
armchair-expert is a chatbot inspired by old Markov chain IRC bots like PyBorg. It regurgitates what it learns from you in unintentionally hilarious ways.

## Features
- Uses NLP to select the most optimal subjects for which to generate a response
- Uses a Recurrent Neural Network (RNN) to structure and capitalize the output, mimicking sentence structure and capitalization of learned text
- Learns new words in real-time with an n-gram markov chain, which is positionally aware of the distances between different words, creating a more coherent sentence

## Requirements
- 3+ GB of RAM
- python 3.6+
- keras (Tensorflow backend)
- spaCy 2.0.0+
- spacymoji
- numpy
- tweepy
- discord.py
- sqlalchemy

## Setup & Training
- Copy config/armchair_expert.example.py to config/armchair_expert.py
- Copy config/ml.example.py to config/ml.py
- Make sure you have the spacy 'en' dataset downloaded: 'python -m spacy download en'
- I would suggest import some data for training before starting the bot. Here is one example: https://github.com/csvance/armchair-expert/blob/master/scripts/import_text_file.py
- Every time the bot starts it will train on all new data it acquired since it started up last
- The bots sentence structure model is only trained once on initial startup. To train it with the most recent acquired data, start the bot with the --retrain-structure flag. If you are noticing the bot is not generating sentences which the structure of learned material, this will help.

# Connectors
## Twitter
- You will need to create an application on the twitter devleoper site on your bot's twitter account https://apps.twitter.com
- After creating it, assign it permissions to do direct messages (this isn't default)
- Create an access token for your account
- Copy config/twitter.example.py to config/twitter.py
- Fill in the tokens and secrets along with your handle
- python armchair_expert.py

## Discord
- You will need to register a bot with Discord: https://discordapp.com/developers/applications/me#top
- Once you register it take note of the Client ID, Username, and Token
- Copy config/discord.example.py to config/discord.py and fill in the relevant fields
- python armchair_expert.py
- When the bot starts you should see a message print to the console containing a link which will allow you to join the bot to a server.

# python-copebot
A machine learning chatbot based off of csvance's armchair-expert (https://github.com/csvance/armchair-expert)

# Windows Setup
Currently, Copebot Python Edition has been tested to work with the following dependencies:
- Python 3.7.7 [LINK](https://www.python.org/downloads/release/python-377/)
- Cuda 10.0.130 [LINK](https://developer.nvidia.com/cuda-10.0-download-archive)
- Cudnn v7.6.5.32 for Cuda 10.0 [LINK](https://developer.nvidia.com/rdp/cudnn-download#a-collapse765-10)

After installing these, use pip to install [requirements.txt](https://gist.githubusercontent.com/collectioncard/ec212a338400b003a72a6ac7d75d3fc7/raw/c7e354204dcaa59f458b8beff5f24f460d9632bb/requirements.txt) via the command ``pip install --no-cache-dir -r requirements.txt``

Finally, install spacy with the command ``python -m spacy download en_core_web_sm``

Now that all of the requirements are installed, add your discord bot information to the bot_config.py file under /config

You should now be able to run copebot by double clicking the copebot_python_edition.py file

\*NOTE: If you have multiple versions of python installed on your system and python 3.7 is not your default, you can run copebot with the correct version of python using the [python launcher](https://docs.python.org/3/using/windows.html#launcher). CD into the same folder as the .py file and run   
`py -3.7 copebot_python_edition.py`

# Mac OS Setup (Note: No GPU Support)
Currently, Copebot Python Edition requires Python 3.7.7 [LINK](https://www.python.org/downloads/release/python-377/)

After installing python, use pip to install [requirements.txt](https://gist.githubusercontent.com/collectioncard/130e0fa0c626020e32611c1c8d18366a/raw/0f05c35f71b6641785b2f93c266803e382f378ac/requirements.txt) via the command ``pip3 install --no-cache-dir -r requirements.txt``

Finally, install spacy with the command ``python3 -m spacy download en_core_web_sm``

Now that all of the requirements are installed, add your discord bot information to the bot_config.py file under /config
   
   Next, set "USE_GPU" to false in ai_config.py

You should now be able to cd into the copebot folder and launch it via the command `python3 copebot_python_edition.py`

\*NOTE: NOTE: If you get a certificate error, run the 'install certificates.command' file located in the python folder in the applications folder. 



##Having issues running copebot?
Try updating your version of discord.py to the latest version of discord.py. (check to make sure the version you are running is at least the same as the one listed in your systems requirements.txt file)

## License
The source code in this repository is licenced under the [AGPL-3.0 license](LICENSE), with portions code licensed under the MIT License, which is as follows:
```
MIT License

Copyright (c) 2017 Carroll Vance

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
