"""Chat Voice - hold a key, talk, and your words land on the clipboard ready for game chat."""
import threading
import queue
import time
import re
import os
import json
import shutil
import site
import difflib
import winsound
from pathlib import Path
from tkinter import messagebox
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import customtkinter as ctk


# ---- let Windows find the NVIDIA libraries installed by pip ----
def add_nvidia_dll_paths():
    try:
        bases = site.getsitepackages() + [site.getusersitepackages()]
    except Exception:
        return
    for base in bases:
        for sub in ("cublas", "cudnn"):
            folder = Path(base) / "nvidia" / sub / "bin"
            if folder.exists():
                try:
                    os.add_dll_directory(str(folder))
                except (OSError, AttributeError):
                    pass
                os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")


add_nvidia_dll_paths()

import ctranslate2
from faster_whisper import WhisperModel

# ---- settings you might change ----
HOTKEY = "f9"          # hold to talk, release to transcribe
CYCLE_HOTKEY = "f10"   # press to switch to the next channel
CHAT_LIMIT = 255       # game's chat limit, counting the channel prefix
SAMPLE_RATE = 16000
MAX_PROMPT_WORDS = 150 # Whisper only reads roughly this much of the word list
MAX_CARDS = 150        # how many history lines to show
MACRO_WORD = "macro"   # say this first, then a macro name, to run a macro

DEVICE_CHOICES = ["Auto", "GPU", "CPU"]
MODEL_CHOICES = ["tiny.en", "base.en", "small.en", "large-v3-turbo"]

# ---- look ----
FONT = "Segoe UI"
BG = "#161b26"          # window
PANEL = "#1f2636"       # text box and side panel
PANEL_HI = "#2a3346"    # cards and quiet buttons
HOVER = "#36415a"
TEXT = "#e8ecf3"
MUTED = "#8b95a8"
TOO_LONG_FILL = "#3b2127"
STATE_COLORS = {
    "idle": "#3a4458",
    "loading": "#6b7a99",
    "listening": "#e5484d",
    "working": "#e9a23b",
    "ready": "#3fb96b",
    "error": "#e5484d",
}
# channel pills borrow the idea of MMO chat colours: each channel gets its own
NONE_COLOR = "#4a5468"
CHANNEL_COLORS = ["#2f9e44", "#3b6fd8", "#c2419a", "#b7791f", "#7c4dcc", "#d9622b"]

# each sound is a list of (pitch in Hz, length in milliseconds); pitch 0 = a pause
SOUND_TONES = {
    "start": [(880, 60)],
    "ready": [(660, 70), (990, 90)],
    "error": [(300, 180)],
    "too_long": [(300, 120), (0, 80), (300, 120)],
    "switch": [(990, 50), (0, 40), (990, 50)],     # two quick blips: channel switched by voice
}

APP_DIR = Path(__file__).parent
HISTORY_DIR = APP_DIR / "chat_history"
GAMES_DIR = APP_DIR / "games"
SETTINGS_FILE = APP_DIR / "settings.json"
OLD_CHANNELS_FILE = APP_DIR / "channels.json"  # from an earlier version
HISTORY_DIR.mkdir(exist_ok=True)
GAMES_DIR.mkdir(exist_ok=True)

WOW_STARTER_WORDS = """# One word or name per line. Lines starting with # are ignored.
# Keep this to words you actually say. Friends' names and guild name matter most.
LFG
LFM
WTS
WTB
DPS
tank
healer
aggro
AoE
respec
pally
warlock
Orgrimmar
Stormwind
Ironforge
Undercity
Thunder Bluff
Darnassus
Auction House
"""

FIXES_STARTER = """# Fixes for words Whisper gets wrong the same way every time.
# Format:  what it wrote = what you meant
# Example:
# or grim mar = Orgrimmar
"""

WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}

frames = []
recording = False
results = queue.Queue()

# speech engine state
model = None
model_lock = threading.Lock()
engine_ready = False
engine_loading = False
loaded_choice = (None, None)

# screen state
current_state = "loading"
state_token = 0
too_long = False
last_spoken = ""         # last thing you said, without prefix
last_message_shown = ""  # what the app last put in the box and on the clipboard
history_cards = []
empty_label = None
side_mode = "history"


def gpu_count():
    try:
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def safe_name(text, fallback):
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', "", text)
    name = name.strip().rstrip(".")
    if not name or name.lower() in WINDOWS_RESERVED:
        name = fallback
    return name


# ---- saved settings ----
def load_settings():
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


saved = load_settings()
start_device = saved.get("device", "Auto")
if start_device not in DEVICE_CHOICES:
    start_device = "Auto"
start_model = saved.get("model", "large-v3-turbo" if gpu_count() > 0 else "base.en")
if start_model not in MODEL_CHOICES:
    start_model = "base.en"
SOUNDS = bool(saved.get("sounds", True))
AUTO_PASTE = bool(saved.get("auto_paste", False))


def save_settings():
    data = {
        "last_game": current_game,
        "device": device_menu.get(),
        "model": model_menu.get(),
        "sounds": SOUNDS,
        "auto_paste": AUTO_PASTE,
    }
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---- sounds ----
def play_sound(name):
    if not SOUNDS:
        return

    def run():
        for pitch, length in SOUND_TONES[name]:
            if pitch == 0:
                time.sleep(length / 1000)
                continue
            try:
                winsound.Beep(pitch, length)
            except RuntimeError:
                pass

    threading.Thread(target=run, daemon=True).start()


# ---- games ----
def game_dir(game):
    return GAMES_DIR / game


def ensure_game(game, starter_words=""):
    folder = game_dir(game)
    folder.mkdir(exist_ok=True)
    words = folder / "words.txt"
    fixes = folder / "fixes.txt"
    channels_file = folder / "channels.json"
    if not words.exists():
        words.write_text(starter_words or "# One word or name per line.\n", encoding="utf-8")
    if not fixes.exists():
        fixes.write_text(FIXES_STARTER, encoding="utf-8")
    if not channels_file.exists():
        channels_file.write_text("[]", encoding="utf-8")


def list_games():
    return sorted(p.name for p in GAMES_DIR.iterdir() if p.is_dir())


def load_words(game):
    try:
        lines = (game_dir(game) / "words.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    words = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.extend(part.strip() for part in line.split(",") if part.strip())
    return words


def load_fixes(game):
    try:
        lines = (game_dir(game) / "fixes.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    fixes = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        wrong, right = line.split("=", 1)
        if wrong.strip():
            fixes.append((wrong.strip(), right.strip()))
    return fixes


def apply_fixes(text, fixes):
    for wrong, right in sorted(fixes, key=lambda pair: len(pair[0]), reverse=True):
        text = re.sub(r"\b" + re.escape(wrong) + r"\b", right, text, flags=re.IGNORECASE)
    return text


if not list_games():
    ensure_game("WoW", WOW_STARTER_WORDS)
    if OLD_CHANNELS_FILE.exists():
        shutil.move(str(OLD_CHANNELS_FILE), str(game_dir("WoW") / "channels.json"))

current_game = saved.get("last_game", "")
if current_game not in list_games():
    current_game = list_games()[0]
ensure_game(current_game)


# ---- channels ----
def load_channels(game):
    try:
        data = json.loads((game_dir(game) / "channels.json").read_text(encoding="utf-8"))
        return [str(p).strip() for p in data if str(p).strip()]
    except (OSError, ValueError):
        return []


def save_channels():
    (game_dir(current_game) / "channels.json").write_text(json.dumps(channels, indent=2), encoding="utf-8")


channels = load_channels(current_game)
current_prefix = ""
selected_index = 0
channel_buttons = []


def build_message(text):
    if current_prefix:
        return f"{current_prefix} {text}"
    return text


def channel_name():
    return current_prefix or "no channel"


# Whisper writes numbers either way, and mishears some of them, so only right after "slash"
# these words count as numbers. "I want to go" elsewhere is untouched.
NUMBER_WORDS = {
    "one": "1", "won": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3",
    "four": "4", "for": "4", "fore": "4",
    "five": "5", "six": "6", "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9", "ten": "10",
}

SLASH_PATTERN = re.compile(
    r"^\s*(?:slash\b[\s,.:;!?-]*|/\s*)([A-Za-z0-9]+)[\s,.:;!?-]*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


def parse_slash_command(text):
    """Look at the very first word only.
    "slash two"              -> ("/2", "")             switch channel
    "Slash 2, anyone group?" -> ("/2", "anyone group?") one line on /2
    "I said slash before"    -> (None, text)           normal message
    """
    match = SLASH_PATTERN.match(text)
    if not match:
        return None, text
    word = match.group(1).lower()
    channel = NUMBER_WORDS.get(word, word)
    channel = SLASH_ALIASES.get(channel, channel)
    rest = match.group(2).strip()
    return "/" + channel, rest


# Words that mean a real chat channel. These get a button and can be switched to.
# Anything else after "slash" (afk, dnd, roll, dance...) just runs once and never sticks.
# Based on WoW's usual chat commands.
CHANNEL_WORDS = {
    "g", "guild", "o", "officer",
    "p", "party", "raid", "ra", "rw",
    "i", "instance", "bg",
    "s", "say", "y", "yell",
    "r", "tt",
}

# Easier things to say that turn into the real command.
SLASH_ALIASES = {
    "target": "tt",   # "slash target" = tell target (plain /target would pick a target instead)
    "reply": "r",
}


def is_channel_prefix(prefix):
    name = prefix[1:]
    return name.isdigit() or name in CHANNEL_WORDS


# ---- macros ----
MACRO_PATTERN = re.compile(r"^\s*" + re.escape(MACRO_WORD) + r"\b[\s,.:;!?-]*(.*)$", re.IGNORECASE | re.DOTALL)


def normalize_name(text):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def load_macros(game):
    try:
        data = json.loads((game_dir(game) / "macros.json").read_text(encoding="utf-8"))
        return [m for m in data if isinstance(m, dict) and str(m.get("name", "")).strip()]
    except (OSError, ValueError):
        return []


def save_macros(game, macros):
    (game_dir(game) / "macros.json").write_text(json.dumps(macros, indent=2), encoding="utf-8")


def match_macro(text, game):
    """None if the line doesn't start with the macro word.
    (True, macro) if a macro matched. (False, what_was_heard) if not."""
    match = MACRO_PATTERN.match(text)
    if not match:
        return None
    heard = normalize_name(match.group(1))
    if not heard:
        return (False, "")
    by_name = {normalize_name(m["name"]): m for m in load_macros(game)}
    if heard in by_name:
        return (True, by_name[heard])
    close = difflib.get_close_matches(heard, list(by_name), n=1, cutoff=0.75)
    if close:
        return (True, by_name[close[0]])
    return (False, heard)


def channel_color(index):
    if index == 0:
        return NONE_COLOR
    return CHANNEL_COLORS[(index - 1) % len(CHANNEL_COLORS)]


# ---- history files ----
def make_file_path(text):
    name = safe_name(text[:10], "note")
    path = HISTORY_DIR / f"{name}.txt"
    number = 2
    while path.exists():
        path = HISTORY_DIR / f"{name} ({number}).txt"
        number += 1
    return path


def save_to_history(text):
    path = make_file_path(text)
    path.write_text(text, encoding="utf-8")
    add_history_card(text, path)


# ---- speech engine ----
def try_load(device, model_name):
    compute = "float16" if device == "cuda" else "int8"
    new_model = WhisperModel(model_name, device=device, compute_type=compute)
    # warm-up on one second of silence: some GPU problems only show up on first use
    segments, _ = new_model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")
    list(segments)
    return new_model


def load_engine_worker(device_choice, model_name):
    global model, engine_ready, engine_loading, loaded_choice
    results.put(("engine_loading", f"Loading {model_name}. The first time, it downloads."))
    new_model = None
    note = ""
    desc = ""

    if device_choice in ("Auto", "GPU"):
        try:
            if gpu_count() == 0:
                raise RuntimeError("no NVIDIA card found")
            new_model = try_load("cuda", model_name)
            desc = f"Running on GPU with {model_name}"
        except Exception as error:
            print("GPU load failed:", error)
            note = "GPU failed, so "

    if new_model is None:
        try:
            new_model = try_load("cpu", model_name)
            desc = f"{note}running on CPU with {model_name}"
            desc = desc[0].upper() + desc[1:]
            if model_name == "large-v3-turbo":
                desc += ". Expect it to be slow."
        except Exception as error:
            print("CPU load failed:", error)
            engine_loading = False
            engine_ready = model is not None
            play_sound("error")
            results.put(("engine_done", (f"Could not load {model_name}. Pick another model in Settings.", False)))
            return

    with model_lock:
        model = new_model
    loaded_choice = (device_choice, model_name)
    engine_ready = True
    engine_loading = False
    play_sound("ready")
    results.put(("engine_done", (desc, True)))


def start_engine_load(device_choice, model_name):
    global engine_loading, engine_ready
    engine_loading = True
    engine_ready = False
    threading.Thread(target=load_engine_worker, args=(device_choice, model_name), daemon=True).start()


# ---- recording (runs on the keyboard thread, so it only talks to the window through the queue) ----
def audio_callback(indata, frame_count, time_info, status):
    if recording:
        frames.append(indata.copy())


def start_recording(event=None):
    global recording, frames
    if recording:
        return
    if not engine_ready:
        play_sound("error")
        results.put(("state", ("error", "The speech model is still loading.")))
        return
    frames = []
    recording = True
    play_sound("start")
    results.put(("state", ("listening", "Listening")))


def stop_recording(event=None):
    global recording
    if not recording:
        return
    recording = False
    if frames:
        audio = np.concatenate(frames).flatten()
    else:
        audio = np.zeros(0, dtype=np.float32)
    threading.Thread(target=transcribe, args=(audio, current_game), daemon=True).start()


def transcribe(audio, game):
    if len(audio) < SAMPLE_RATE * 0.3:
        play_sound("error")
        results.put(("state", ("error", "Too short. Hold the key a little longer.")))
        return
    with model_lock:
        current_model = model
    if current_model is None:
        play_sound("error")
        results.put(("state", ("error", "No speech model loaded. Pick one in Settings.")))
        return
    results.put(("state", ("working", "Working")))
    macro_names = [m["name"] for m in load_macros(game)]
    words = (["slash", MACRO_WORD] + macro_names + load_words(game))[:MAX_PROMPT_WORDS]
    prompt = ("Glossary: " + ", ".join(words) + ".") if words else None
    segments, _ = current_model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                           initial_prompt=prompt)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    text = apply_fixes(text, load_fixes(game))
    text = text.rstrip(".").strip()
    if not text:
        play_sound("error")
        results.put(("state", ("error", "Didn't catch that. Try again.")))
        return
    results.put(("text", text))


def auto_paste():
    time.sleep(0.1)
    keyboard.send("enter")
    time.sleep(0.15)
    keyboard.send("ctrl+v")
    time.sleep(0.1)
    keyboard.send("enter")


# ---- window behaviour ----
def border_for(state):
    if state in ("listening", "working", "loading"):
        return STATE_COLORS[state]
    if too_long:
        return STATE_COLORS["error"]
    return STATE_COLORS[state]


def set_state(state, text=None):
    global current_state, state_token
    current_state = state
    state_token += 1
    status_dot.configure(text_color=STATE_COLORS[state])
    if text is not None:
        status_label.configure(text=text)
    text_box.configure(border_color=border_for(state))
    if state in ("ready", "error"):
        token = state_token
        root.after(4000, lambda: fade_to_idle(token))


def fade_to_idle(token):
    global current_state
    if token != state_token:
        return  # something newer happened
    current_state = "idle"
    status_dot.configure(text_color=STATE_COLORS["idle"])
    text_box.configure(border_color=border_for("idle"))


def check_length(message=None):
    """Update the counter and colours. Returns True if the message fits."""
    global too_long
    if message is None:
        message = text_box.get("1.0", "end").strip()
    length = len(message)
    fits = length <= CHAT_LIMIT
    too_long = not fits
    text_box.configure(fg_color=PANEL if fits else TOO_LONG_FILL)
    counter_label.configure(text=f"{length} / {CHAT_LIMIT}", text_color=MUTED if fits else "#ff7b7b")
    text_box.configure(border_color=border_for(current_state))
    return fits


def show_and_copy(message):
    global last_message_shown
    text_box.delete("1.0", "end")
    text_box.insert("1.0", message)
    pyperclip.copy(message)
    last_message_shown = message
    return check_length(message)


def deliver(text, from_history=False, prefix=None, remember=True):
    """Put a line on the clipboard and report how it went.
    prefix=None uses the selected channel; a prefix like "/2" sends just this line there.
    remember=False is for one-off commands, so F10 won't try to re-send them."""
    global last_spoken
    last_spoken = text if remember else ""
    use_prefix = current_prefix if prefix is None else prefix
    message = f"{use_prefix} {text}".strip() if use_prefix else text
    name = use_prefix or "no channel"
    if show_and_copy(message):
        play_sound("ready")
        if AUTO_PASTE and not from_history:
            set_state("ready", f"Sent to {name}")
            threading.Thread(target=auto_paste, daemon=True).start()
        else:
            set_state("ready", f"Ready for {name}. In game: Enter, Ctrl+V, Enter")
    else:
        play_sound("too_long")
        set_state("error", f"Too long by {len(message) - CHAT_LIMIT}. Trim it, then copy.")


def send_lines(lines):
    """Type each line into the game chat in turn: Enter, paste, Enter."""
    for line in lines:
        pyperclip.copy(line)
        time.sleep(0.05)
        keyboard.send("enter")
        time.sleep(0.15)
        keyboard.send("ctrl+v")
        time.sleep(0.1)
        keyboard.send("enter")
        time.sleep(0.3)


def run_macro(macro):
    global last_spoken, last_message_shown
    lines = [line.strip() for line in macro["text"].splitlines() if line.strip()]
    if not lines:
        play_sound("error")
        set_state("error", f"Macro {macro['name']} is empty")
        return
    longest = max(lines, key=len)
    text_box.delete("1.0", "end")
    text_box.insert("1.0", "\n".join(lines))
    last_spoken = ""
    last_message_shown = text_box.get("1.0", "end").strip()
    if not check_length(longest):
        play_sound("too_long")
        set_state("error", f"Macro {macro['name']} has a line over {CHAT_LIMIT}. Not sent.")
        return
    play_sound("ready")
    set_state("ready", f"Running macro {macro['name']}")
    # macros always send themselves: the text is fixed, so there's nothing to check first
    threading.Thread(target=send_lines, args=(lines,), daemon=True).start()


def handle_spoken(value):
    found = match_macro(value, current_game)
    if found is not None:
        ok, item = found
        if ok:
            run_macro(item)
        elif item:
            play_sound("error")
            set_state("error", f'No macro called "{item}"')
        else:
            play_sound("error")
            set_state("error", "Heard macro but no name. Try again.")
        return

    if re.fullmatch(r"\W*slash\W*", value, re.IGNORECASE):
        play_sound("error")
        set_state("error", "Heard slash but no channel. Try again.")
        return

    prefix, rest = parse_slash_command(value)
    if prefix is None:
        save_to_history(value)
        deliver(value)
    elif is_channel_prefix(prefix):
        if not rest:
            # "slash two" on its own: switch the selected channel and stay there
            select_channel(ensure_channel(prefix))
            play_sound("switch")
            set_state("ready", f"Switched to {prefix}")
        else:
            # "slash two, hello": just this line on /2, selected channel stays put
            ensure_channel(prefix)
            save_to_history(rest)
            deliver(rest, prefix=prefix)
    else:
        # a command like /afk or /roll: runs once, no button, not saved to history
        deliver(rest, prefix=prefix, remember=False)


def copy_edited():
    global last_message_shown
    edited = text_box.get("1.0", "end").strip()
    pyperclip.copy(edited)
    last_message_shown = edited
    if check_length(edited):
        set_state("ready", "Copied your edited text")
    else:
        play_sound("too_long")
        set_state("error", f"Copied, but still {len(edited) - CHAT_LIMIT} too long")


# ---- channels on screen ----
def rebuild_channel_buttons():
    global channel_buttons
    for widget in channel_bar.winfo_children():
        widget.destroy()
    channel_buttons = []
    for i, label in enumerate(["None"] + channels):
        color = channel_color(i)
        button = ctk.CTkButton(
            channel_bar, text=label, font=F_BODY, width=56, height=32, corner_radius=16,
            border_width=2, border_color=color, hover_color=color,
            command=lambda idx=i: select_channel(idx),
        )
        button.pack(side="left", padx=(0, 8))
        if i > 0:
            button.bind("<Button-3>", lambda event, idx=i: remove_channel(idx))
        channel_buttons.append(button)
    ctk.CTkButton(
        channel_bar, text="+", font=F_BODY, width=32, height=32, corner_radius=16,
        fg_color="transparent", border_width=2, border_color=PANEL_HI, hover_color=PANEL_HI,
        text_color=MUTED, command=add_channel,
    ).pack(side="left")
    update_channel_styles()


def update_channel_styles():
    for i, button in enumerate(channel_buttons):
        if i == selected_index:
            button.configure(fg_color=channel_color(i), text_color="#ffffff")
        else:
            button.configure(fg_color="transparent", text_color=MUTED)


def select_channel(index):
    global selected_index, current_prefix
    selected_index = index
    current_prefix = "" if index == 0 else channels[index - 1]
    update_channel_styles()


def cycle_channel():
    select_channel((selected_index + 1) % (len(channels) + 1))
    # if the box still holds the last line untouched, re-send it on the new channel
    box_text = text_box.get("1.0", "end").strip()
    if last_spoken and box_text == last_message_shown:
        deliver(last_spoken, from_history=True)
    else:
        check_length()
        set_state("idle", f"Channel: {channel_name()}")


def ask_text(title, prompt):
    dialog = ctk.CTkInputDialog(title=title, text=prompt)
    try:
        dialog.attributes("-topmost", True)  # the main window stays on top, so this has to as well
    except Exception:
        pass
    return dialog.get_input()


def add_channel():
    prefix = ask_text("Add a channel", "Type the chat prefix, like /g or /1 or /w Name")
    if prefix is None:
        return
    prefix = prefix.strip()
    if not prefix:
        return
    select_channel(ensure_channel(prefix))


def ensure_channel(prefix):
    """Add the channel as a button if it's new. Returns its button number."""
    if prefix not in channels:
        channels.append(prefix)
        save_channels()
        rebuild_channel_buttons()
    return channels.index(prefix) + 1


def remove_channel(index):
    global selected_index
    prefix = channels[index - 1]
    if not messagebox.askyesno("Remove channel", f"Remove the channel {prefix}?", parent=root):
        return
    channels.pop(index - 1)
    save_channels()
    if selected_index == index:
        selected_index = 0
    elif selected_index > index:
        selected_index -= 1
    rebuild_channel_buttons()
    select_channel(selected_index)


# ---- history on screen ----
def show_empty_label():
    global empty_label
    if empty_label is None:
        empty_label = ctk.CTkLabel(history_scroll,
                                   text=f"Hold {HOTKEY.upper()} and talk.\nWhat you say shows up here.",
                                   font=F_SMALL, text_color=MUTED, justify="left")
        empty_label.pack(anchor="w", padx=8, pady=8)


def add_history_card(text, path):
    global empty_label
    if empty_label is not None:
        empty_label.destroy()
        empty_label = None
    shown = text if len(text) <= 32 else text[:32] + "..."

    # a card is a frame holding two buttons: the line itself, and an X to delete it
    card = ctk.CTkFrame(history_scroll, fg_color=PANEL_HI, corner_radius=8)
    ctk.CTkButton(
        card, text=shown, anchor="w", font=F_SMALL, height=34, corner_radius=8,
        fg_color="transparent", hover_color=HOVER, text_color=TEXT,
        command=lambda t=text: deliver(t, from_history=True),
    ).pack(side="left", fill="x", expand=True)
    ctk.CTkButton(
        card, text="×", font=ctk.CTkFont(family=FONT, size=16), width=30, height=34, corner_radius=8,
        fg_color="transparent", hover_color="#4a2a30", text_color=MUTED,
        command=lambda: delete_history_card(card, path),
    ).pack(side="right")

    if history_cards:
        card.pack(fill="x", padx=4, pady=3, before=history_cards[0])
    else:
        card.pack(fill="x", padx=4, pady=3)
    history_cards.insert(0, card)
    if len(history_cards) > MAX_CARDS:
        history_cards.pop().destroy()  # only hides the oldest card; its file stays


def delete_history_card(card, path):
    try:
        path.unlink(missing_ok=True)  # removes the saved file for good
    except OSError as error:
        set_state("error", f"Couldn't delete that file: {error}")
        return
    card.destroy()
    if card in history_cards:
        history_cards.remove(card)
    if not history_cards:
        show_empty_label()


def load_history():
    files = sorted(HISTORY_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime)
    for f in files[-MAX_CARDS:]:
        try:
            add_history_card(f.read_text(encoding="utf-8").strip(), f)
        except OSError:
            pass


def toggle_side():
    global side_mode
    if side_mode == "history":
        history_scroll.grid_forget()
        settings_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0, 8))
        side_title.configure(text="Settings")
        side_toggle.configure(text="Done")
        side_mode = "settings"
    else:
        settings_frame.grid_forget()
        history_scroll.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0, 8))
        side_title.configure(text="History")
        side_toggle.configure(text="Settings")
        side_mode = "history"


# ---- settings on screen ----
def switch_game(game):
    global current_game, channels
    current_game = game
    ensure_game(game)
    channels = load_channels(game)
    rebuild_channel_buttons()
    select_channel(0)
    game_menu.set(game)
    save_settings()
    set_state("idle", f"Playing {game}")


def add_game():
    name = ask_text("Add a game", "Name of the game")
    if name is None:
        return
    name = safe_name(name, "")
    if not name:
        return
    ensure_game(name)
    game_menu.configure(values=list_games())
    switch_game(name)


def open_list(filename):
    ensure_game(current_game)
    os.startfile(game_dir(current_game) / filename)  # save it and the next line you say uses it


def on_engine_change(_choice=None):
    if engine_loading:
        device_menu.set(loaded_choice[0] or start_device)
        model_menu.set(loaded_choice[1] or start_model)
        set_state("error", "Still loading. Try again in a moment.")
        return
    if (device_menu.get(), model_menu.get()) == loaded_choice:
        return
    save_settings()
    start_engine_load(device_menu.get(), model_menu.get())


def toggle_sounds():
    global SOUNDS
    SOUNDS = bool(sound_switch.get())
    save_settings()


def toggle_paste():
    global AUTO_PASTE
    AUTO_PASTE = bool(paste_switch.get())
    save_settings()


macro_window = None


def open_macro_editor():
    global macro_window
    if macro_window is not None and macro_window.winfo_exists():
        macro_window.lift()
        macro_window.focus()
        return

    game = current_game
    macros = load_macros(game)
    editing = {"index": None}

    win = ctk.CTkToplevel(root, fg_color=BG)
    macro_window = win
    win.title(f"Macros for {game}")
    win.geometry("660x380")
    win.attributes("-topmost", True)
    win.after(200, win.lift)
    win.grid_columnconfigure(1, weight=1)
    win.grid_rowconfigure(0, weight=1)

    # left: list of macros
    left_panel = ctk.CTkFrame(win, fg_color=PANEL, corner_radius=12, width=200)
    left_panel.grid(row=0, column=0, sticky="ns", padx=(16, 8), pady=16)
    left_panel.grid_propagate(False)
    left_panel.grid_columnconfigure(0, weight=1)
    left_panel.grid_rowconfigure(1, weight=1)
    ctk.CTkLabel(left_panel, text=game, font=F_HEAD, text_color=TEXT).grid(
        row=0, column=0, sticky="w", padx=12, pady=(10, 4))
    name_list = ctk.CTkScrollableFrame(left_panel, fg_color="transparent")
    name_list.grid(row=1, column=0, sticky="nsew", padx=4)
    quiet_button(left_panel, "New macro", lambda: start_new(), width=170).grid(row=2, column=0, pady=10)

    # right: the macro being edited
    right_panel = ctk.CTkFrame(win, fg_color="transparent")
    right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
    right_panel.grid_columnconfigure(0, weight=1)
    right_panel.grid_rowconfigure(3, weight=1)
    ctk.CTkLabel(right_panel, text=f"Name. You'll say: {MACRO_WORD} and then this name.",
                 font=F_SMALL, text_color=MUTED).grid(row=0, column=0, sticky="w")
    name_entry = ctk.CTkEntry(right_panel, font=F_BODY, height=34, fg_color=PANEL,
                              border_color=PANEL_HI, text_color=TEXT)
    name_entry.grid(row=1, column=0, sticky="ew", pady=(2, 10))
    ctk.CTkLabel(right_panel, text="What it sends. One command per line.",
                 font=F_SMALL, text_color=MUTED).grid(row=2, column=0, sticky="w")
    body = ctk.CTkTextbox(right_panel, font=F_BODY, wrap="none", fg_color=PANEL, text_color=TEXT,
                          border_width=2, border_color=PANEL_HI, corner_radius=8)
    body.grid(row=3, column=0, sticky="nsew", pady=(2, 10))

    bottom = ctk.CTkFrame(right_panel, fg_color="transparent")
    bottom.grid(row=4, column=0, sticky="ew")
    info = ctk.CTkLabel(bottom, text="", font=F_SMALL, text_color=MUTED, wraplength=260, justify="left")
    info.pack(side="left")
    ctk.CTkButton(bottom, text="Delete", font=F_SMALL, width=80, height=30, corner_radius=15,
                  fg_color="transparent", border_width=2, border_color="#4a2a30",
                  hover_color="#4a2a30", text_color=TEXT, command=lambda: delete()).pack(side="right", padx=(8, 0))
    ctk.CTkButton(bottom, text="Save", font=F_SMALL, width=80, height=30, corner_radius=15,
                  fg_color=STATE_COLORS["ready"], hover_color="#2f9e5a", text_color="#ffffff",
                  command=lambda: save()).pack(side="right")

    def refresh_list():
        for widget in name_list.winfo_children():
            widget.destroy()
        if not macros:
            ctk.CTkLabel(name_list, text="No macros yet.", font=F_SMALL, text_color=MUTED).pack(
                anchor="w", padx=8, pady=6)
        for i, item in enumerate(macros):
            ctk.CTkButton(
                name_list, text=item["name"], anchor="w", font=F_SMALL, height=30, corner_radius=8,
                fg_color=PANEL_HI if i == editing["index"] else "transparent",
                hover_color=HOVER, text_color=TEXT, command=lambda idx=i: load_into(idx),
            ).pack(fill="x", padx=2, pady=2)

    def load_into(index):
        editing["index"] = index
        item = macros[index]
        name_entry.delete(0, "end")
        name_entry.insert(0, item["name"])
        body.delete("1.0", "end")
        body.insert("1.0", item["text"])
        info.configure(text=f"Say: {MACRO_WORD} {item['name']}")
        refresh_list()

    def start_new():
        editing["index"] = None
        name_entry.delete(0, "end")
        body.delete("1.0", "end")
        info.configure(text="New macro")
        refresh_list()
        name_entry.focus()

    def save():
        name = name_entry.get().strip()
        text = body.get("1.0", "end").strip()
        if not name or not text:
            info.configure(text="Give it a name and something to send.")
            return
        key = normalize_name(name)
        if not key:
            info.configure(text="The name needs some letters or numbers.")
            return
        for j, item in enumerate(macros):
            if j != editing["index"] and normalize_name(item["name"]) == key:
                info.configure(text="Another macro already has that name.")
                return
        new_item = {"name": name, "text": text}
        if editing["index"] is None:
            macros.append(new_item)
            editing["index"] = len(macros) - 1
        else:
            macros[editing["index"]] = new_item
        save_macros(game, macros)
        refresh_list()
        note = f"Saved. Say: {MACRO_WORD} {name}"
        if any(len(line.strip()) > CHAT_LIMIT for line in text.splitlines()):
            note += f". Warning: a line is over {CHAT_LIMIT} characters and won't send."
        info.configure(text=note)

    def delete():
        if editing["index"] is None:
            start_new()
            return
        name = macros[editing["index"]]["name"]
        if not messagebox.askyesno("Delete macro", f"Delete the macro {name}?", parent=win):
            return
        macros.pop(editing["index"])
        save_macros(game, macros)
        start_new()

    start_new()


def section_label(parent, text):
    ctk.CTkLabel(parent, text=text, font=F_SMALL, text_color=MUTED).pack(anchor="w", padx=8, pady=(12, 4))


def quiet_button(parent, text, command, width=120):
    return ctk.CTkButton(parent, text=text, font=F_SMALL, width=width, height=30, corner_radius=15,
                         fg_color=PANEL_HI, hover_color=HOVER, text_color=TEXT, command=command)


def poll_results():
    while not results.empty():
        kind, value = results.get()
        if kind == "state":
            set_state(*value)
        elif kind == "engine_loading":
            engine_label.configure(text=value)
            set_state("loading", "Loading speech model")
        elif kind == "engine_done":
            text, ok = value
            engine_label.configure(text=text)
            if ok:
                set_state("ready", f"Ready. Hold {HOTKEY.upper()} and talk.")
            else:
                set_state("error", text)
        elif kind == "cycle":
            cycle_channel()
        elif kind == "text":
            handle_spoken(value)
    root.after(100, poll_results)


# ================= build the window =================
ctk.set_appearance_mode("dark")
root = ctk.CTk(fg_color=BG)
root.title("Chat Voice")
root.attributes("-topmost", True)
root.geometry("960x360")
root.minsize(780, 320)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

F_BODY = ctk.CTkFont(family=FONT, size=13)
F_SMALL = ctk.CTkFont(family=FONT, size=12)
F_TEXT = ctk.CTkFont(family=FONT, size=17)
F_HEAD = ctk.CTkFont(family=FONT, size=14, weight="bold")

# main column
main = ctk.CTkFrame(root, fg_color="transparent")
main.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
main.grid_columnconfigure(0, weight=1)
main.grid_rowconfigure(1, weight=1)

channel_bar = ctk.CTkFrame(main, fg_color="transparent")
channel_bar.grid(row=0, column=0, sticky="ew", pady=(0, 12))

text_box = ctk.CTkTextbox(main, font=F_TEXT, wrap="word", fg_color=PANEL, text_color=TEXT,
                          border_width=2, border_color=STATE_COLORS["loading"], corner_radius=12)
text_box.grid(row=1, column=0, sticky="nsew")
text_box.bind("<KeyRelease>", lambda event: check_length())

status_row = ctk.CTkFrame(main, fg_color="transparent")
status_row.grid(row=2, column=0, sticky="ew", pady=(12, 0))
status_dot = ctk.CTkLabel(status_row, text="●", width=16, font=ctk.CTkFont(family=FONT, size=16),
                          text_color=STATE_COLORS["loading"])
status_dot.pack(side="left")
status_label = ctk.CTkLabel(status_row, text="Loading speech model", font=F_BODY, text_color=TEXT)
status_label.pack(side="left", padx=(6, 0))
quiet_button(status_row, "Copy edited text", copy_edited, width=130).pack(side="right")
counter_label = ctk.CTkLabel(status_row, text=f"0 / {CHAT_LIMIT}", font=F_SMALL, text_color=MUTED)
counter_label.pack(side="right", padx=12)

# side column
side = ctk.CTkFrame(root, fg_color=PANEL, corner_radius=12, width=300)
side.grid(row=0, column=1, sticky="ns", padx=(8, 16), pady=16)
side.grid_propagate(False)
side.grid_columnconfigure(0, weight=1)
side.grid_rowconfigure(1, weight=1)

side_title = ctk.CTkLabel(side, text="History", font=F_HEAD, text_color=TEXT)
side_title.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 6))
side_toggle = ctk.CTkButton(side, text="Settings", font=F_SMALL, width=80, height=26, corner_radius=13,
                            fg_color=PANEL_HI, hover_color=HOVER, text_color=TEXT, command=toggle_side)
side_toggle.grid(row=0, column=1, sticky="e", padx=10, pady=(10, 6))

history_scroll = ctk.CTkScrollableFrame(side, fg_color="transparent")
history_scroll.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=(0, 8))
show_empty_label()

settings_frame = ctk.CTkScrollableFrame(side, fg_color="transparent")  # shown when you press Settings

section_label(settings_frame, "Game")
game_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
game_row.pack(fill="x", padx=8)
game_menu = ctk.CTkOptionMenu(game_row, values=list_games(), command=switch_game, font=F_SMALL,
                              width=140, height=30, fg_color=PANEL_HI, button_color=PANEL_HI,
                              button_hover_color=HOVER, text_color=TEXT)
game_menu.set(current_game)
game_menu.pack(side="left")
quiet_button(game_row, "Add game", add_game, width=90).pack(side="left", padx=(8, 0))

lists_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
lists_row.pack(fill="x", padx=8, pady=(8, 0))
quiet_button(lists_row, "Edit word list", lambda: open_list("words.txt"), width=115).pack(side="left")
quiet_button(lists_row, "Edit fixes", lambda: open_list("fixes.txt"), width=115).pack(side="left", padx=(8, 0))

section_label(settings_frame, "Macros")
quiet_button(settings_frame, "Edit macros", lambda: open_macro_editor(), width=238).pack(anchor="w", padx=8)

section_label(settings_frame, "Speech engine")
engine_row = ctk.CTkFrame(settings_frame, fg_color="transparent")
engine_row.pack(fill="x", padx=8)
device_menu = ctk.CTkOptionMenu(engine_row, values=DEVICE_CHOICES, command=on_engine_change, font=F_SMALL,
                                width=80, height=30, fg_color=PANEL_HI, button_color=PANEL_HI,
                                button_hover_color=HOVER, text_color=TEXT)
device_menu.set(start_device)
device_menu.pack(side="left")
model_menu = ctk.CTkOptionMenu(engine_row, values=MODEL_CHOICES, command=on_engine_change, font=F_SMALL,
                               width=150, height=30, fg_color=PANEL_HI, button_color=PANEL_HI,
                               button_hover_color=HOVER, text_color=TEXT)
model_menu.set(start_model)
model_menu.pack(side="left", padx=(8, 0))
engine_label = ctk.CTkLabel(settings_frame, text="Starting", font=F_SMALL, text_color=MUTED,
                            wraplength=250, justify="left")
engine_label.pack(anchor="w", padx=8, pady=(6, 0))

section_label(settings_frame, "While playing")
sound_switch = ctk.CTkSwitch(settings_frame, text="Sounds", font=F_SMALL, text_color=TEXT, command=toggle_sounds)
sound_switch.pack(anchor="w", padx=8, pady=4)
if SOUNDS:
    sound_switch.select()
paste_switch = ctk.CTkSwitch(settings_frame, text="Paste into game for me", font=F_SMALL, text_color=TEXT,
                             command=toggle_paste)
paste_switch.pack(anchor="w", padx=8, pady=4)
if AUTO_PASTE:
    paste_switch.select()

# ================= start up =================
rebuild_channel_buttons()
select_channel(0)
load_history()
check_length("")

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_callback)
stream.start()

keyboard.on_press_key(HOTKEY, start_recording)
keyboard.on_release_key(HOTKEY, stop_recording)
keyboard.on_release_key(CYCLE_HOTKEY, lambda event: results.put(("cycle", None)))

start_engine_load(start_device, start_model)

root.after(100, poll_results)
root.mainloop()

stream.stop()
keyboard.unhook_all()
