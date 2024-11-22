# -*- coding: utf-8 -*-
import datetime
import json
import os
import locale
import logging
import re
import sys
from queue import Queue
from pathlib import Path


def get_executable_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable).replace('\\', '/')
    else:
        return str(Path.cwd()).replace('\\', '/')


# root dir
rootdir = get_executable_path()
root_path = Path(rootdir)

# cache tmp
temp_path = root_path / "tmp"
temp_path.mkdir(parents=True, exist_ok=True)
TEMP_DIR = temp_path.as_posix()

# home 
homepath = Path.home() / 'Videos/dub'
homepath.mkdir(parents=True, exist_ok=True)
homedir = homepath.as_posix()

# home tmp
TEMP_HOME = homedir + "/tmp"
Path(TEMP_HOME).mkdir(parents=True, exist_ok=True)

# logs 

logs_path = root_path / "logs"
logs_path.mkdir(parents=True, exist_ok=True)
LOGS_DIR = logs_path.as_posix()

logger = logging.getLogger('dub')

## 

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# logger = logging.getLogger('MyLogger')
logger.setLevel(logging.INFO)  # level

file_handler = logging.FileHandler(f'{rootdir}/logs/video-{datetime.datetime.now().strftime("%Y%m%d")}.log',
                                   encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 只记录ERROR及以上级别的日志
output = open(f'{rootdir}/logs/video-{datetime.datetime.now().strftime("%Y%m%d")}-std.log', "wt")
sys.stdout = output
sys.stderr = output

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # allow（Ctrl+C）quit
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# custom hook
sys.excepthook = log_uncaught_exceptions

# ffmpeg
if sys.platform == 'win32':
    PWD = rootdir.replace('/', '\\')
    os.environ['PATH'] = PWD + f';{PWD}\\ffmpeg;' + os.environ['PATH']

else:
    os.environ['PATH'] = rootdir + f':{rootdir}/ffmpeg:' + os.environ['PATH']

os.environ['QT_API'] = 'pyside6'
os.environ['SOFT_NAME'] = 'dub'
# main win
queue_logs = Queue(1000)
# box win
queuebox_logs = Queue(1000)

# start status
current_status = "stop"

queue_novice = {}

video_cache = {}

canceldown = False

box_trans = "stop"
box_tts = "stop"
box_recogn = 'stop'
separate_status = 'stop'
last_opendir = homedir

exit_soft = False

trans_queue = []

dubb_queue = []

regcon_queue = []

compose_queue = []

unidlist = []

errorlist = {}

video_codec = None

video_min_ms = 50
clone_voicelist = ["clone"]
openaiTTS_rolelist = "alloy,echo,fable,onyx,nova,shimmer"

try:
    defaulelang = locale.getdefaultlocale()[0][:2].lower()
except Exception:
    defaulelang = "zh"


def parse_init():
    default = {
            "lang": "",
            "crf": 13,
            "cuda_qp": False,
            "preset": "slow",
            "ffmpeg_cmd": "",
            "video_codec": 264,
            "model_list": "tiny,tiny.en,base,base.en,small,small.en,medium,medium.en,large-v1,large-v2,large-v3,distil-whisper-small.en,distil-whisper-medium.en,distil-whisper-large-v2,distil-whisper-large-v3",
            "voice_silence": 250,
            "interval_split": 10,
            "trans_thread": 15,
            "retries": 2,
            "dubbing_thread": 5,
            "countdown_sec": 15,
            "backaudio_volume": 0.8,
            "separate_sec": 600,
            "loop_backaudio": True,
            "cuda_com_type": "float32",
            "whisper_threads": 4,
            "whisper_worker": 1,
            "beam_size": 5,
            "best_of": 5,
            "temperature": 0,
            "condition_on_previous_text": False,
            "fontsize": 16,
            "fontname": "黑体",
            "fontcolor": "&hffffff",
            "fontbordercolor": "&h000000",
            "subtitle_bottom": 10,
            "cjk_len": 20,
            "cors_run":True,
            "other_len": 54,
            "zh_hant_s": True
    }
    return default # followings no use
    try: 
        tmpjson = json.load(open(rootdir + "/dub/cfg.json", 'r', encoding='utf-8'))
    except Exception as e:
        print(e)
        raise Exception('dub/cfg.json not found  or  error')
    else:
        settings = {}
        for key, val in tmpjson.items():
            value = str(val).strip()
            if value.isdigit():
                settings[key] = int(value)
            elif re.match(r'^\d*\.\d+$', value):
                settings[key] = float(value)
            elif value.lower() == 'true':
                settings[key] = True
            elif value.lower() == 'false':
                settings[key] = False
            else:
                settings[key] = value.lower() if value else ""
        default.update(settings)
        return default


settings = parse_init()

task_thread = False

edgeTTS_rolelist = None
AzureTTS_rolelist = None

proxy = None

# 配置
params = {
    "source_mp4": "",
    "target_dir": "",

    "source_language": "en",
    "detect_language": "en",

    "target_language": "zh-cn",
    "subtitle_language": "chi",

    "cuda": False,
    "is_separate": False,

    "voice_role": "No",
    "voice_rate": "0",

    "tts_type": "edgeTTS",  # 
    "tts_type_list": ["edgeTTS", 'CosyVoice'],

    "whisper_type": "all",
    "whisper_model": "tiny",
    "model_type": "faster",
    "only_video": False,
    "translate_type": "google",
    "subtitle_type": 0,  # embed soft
    "voice_autorate": False,
    "auto_ajust": True,

    "deepl_authkey": "",
    "deepl_api": "",
    "deeplx_address": "",
    "ott_address": "",

    "elevenlabstts_role": [],
    "elevenlabstts_key": "",

    "clone_api": "",
    "zh_recogn_api": "",

    "ttsapi_url": "",
    "ttsapi_voice_role": "",
    "ttsapi_extra": "dub",

    "trans_api_url": "",
    "trans_secret": ""
}

