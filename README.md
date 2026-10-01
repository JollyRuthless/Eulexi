# Eulexi

Hold a key, talk, and your words land on the clipboard ready to paste into game chat. Built for people who want to keep up in MMO chat but type or spell slower than the conversation moves.

Speech is turned into text on your own computer with [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Nothing is sent to the internet.

## What it does

Hold F9 and talk. Let go, and the text shows up in the window and on your clipboard. In the game, press Enter, Ctrl+V, Enter. Or turn on "Paste into game for me" in Settings and the app presses those keys for you.

Channels sit as coloured buttons along the top. Add one with the + button and type the game's chat prefix, like `/g`, `/1`, or `/w Name`. Press F10 to switch channels without leaving the game. Right-click a channel to remove it.

The border around the text shows what the app is doing: red while listening, amber while working, green when your text is ready. You also hear a blip when it starts listening and a chime when the text is ready.

Chat lines over 255 characters turn the text box red and are never auto-pasted, so nothing goes out cut off.

Everything you say is saved in the history panel. Click a line to copy it again on the current channel, or click the × to delete it.

## Voice commands

Start with the word "slash" to change channel. "Slash two" on its own switches to `/2` and stays there. "Slash two, anyone want to group" sends just that line on `/2`. Real chat channels (numbers, guild, party, raid, say, yell, officer, reply, tell target) get a button. "Slash target" means `/tt`, which whispers whoever you have targeted. Other commands, like "slash afk, back in five" or "slash roll", run once and never become a channel. Only the very first word counts, so "slash" later in a sentence is just a word.

## Macros

Under Settings, Edit macros lets you save named commands for each game, one command per line. Say "macro" and then the name, like "macro low graphics", and the app types each line into game chat for you. Macros always send themselves, whether or not "Paste into game for me" is on, because the text is fixed and there's nothing to check first.

## Teaching it game words

Each game gets its own word list and fixes list, under Settings.

The word list is a hint for the speech engine: names and terms it should expect, like your guild name, friends' characters, and zones. Keep it short. The engine only reads about the first 150 words.

The fixes list catches mistakes the engine makes the same way every time. One per line, written as `what it wrote = what you meant`.

## Requirements

Windows 10 or 11, Python 3.9 or newer, and a microphone.

```
pip install -r requirements.txt
```

### Using an NVIDIA graphics card (optional)

The app runs on the CPU out of the box. With an NVIDIA card it can use the larger, more accurate model and still be fast. Install these as well:

```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

Then pick GPU or Auto under Settings. If the card doesn't work, the app falls back to the CPU and says so in Settings. The error details print in the console window.

## Running

```
python eulexi.py
```

The first time you pick a model, it downloads. The small models are under 200 MB and the large one is about 1.6 GB.

If your game runs as administrator, run Eulexi as administrator too, or the hotkeys won't work while the game has focus.

## Settings you can change in the code

The top of `eulexi.py` has the hotkeys (`HOTKEY`, `CYCLE_HOTKEY`) and the chat limit (`CHAT_LIMIT`).

## Your data

The app creates these next to itself, and they are left out of git on purpose because they hold what you've said and the names you play with:

`chat_history/` holds every line you've said, one file each. `games/` holds each game's channels, word list and fixes. `settings.json` holds your choices.
