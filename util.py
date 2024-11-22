# -*- coding: utf-8 -*-


#coding=utf-8
import whisper
import argparse
import config
import os
import uuid
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip, AudioFileClip

import spacy
from spacy_syllables import SpacySyllables
from tqdm import tqdm
import tempfile
import re
import edge_tts
import asyncio
import librosa
import soundfile as sf
import time

import copy
import hashlib
import math
import platform
import random

import re
import shutil
import subprocess
import sys
import os
from datetime import timedelta
import json
from pathlib import Path

import requests
import config
import time

def set_proxy(set_val=''):
    if set_val == 'del':
        config.proxy = None
        # del
        if os.environ.get('http_proxy'):
            os.environ.pop('http_proxy')
        if os.environ.get('https_proxy'):
            os.environ.pop('https_proxy')
        return None
    if set_val:
        # set 
        if not set_val.startswith("http") and not set_val.startswith('sock'):
            set_val = f"http://{set_val}"
        config.proxy = set_val
        os.environ['http_proxy']=set_val
        os.environ['https_proxy']=set_val
        os.environ['all_proxy']=set_val
        return set_val

    # get proxy
    http_proxy = config.proxy or os.environ.get('http_proxy') or os.environ.get('https_proxy')
    if http_proxy:
        if not http_proxy.startswith("http") and not http_proxy.startswith('sock'):
            http_proxy = f"http://{http_proxy}"
        return http_proxy
    if sys.platform != 'win32':
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as key:

            proxy_enable, _ = winreg.QueryValueEx(key, 'ProxyEnable')
            proxy_server, _ = winreg.QueryValueEx(key, 'ProxyServer')
            if proxy_server:
                if not proxy_server.startswith("http") and not proxy_server.startswith('sock'):
                    proxy_server = "http://" + proxy_server
                try:
                    requests.head(proxy_server, proxies={"http": "", "https": ""})
                except Exception:
                    return None
                return proxy_server
    except Exception as e:
        pass
    return None

spacy_models = {
    "english": "en_core_web_sm",
    "german": "de_core_news_sm",
    "french": "fr_core_news_sm",
    "italian": "it_core_news_sm",
    "catalan": "ca_core_news_sm",
    "chinese": "zh_core_web_sm",
    "croatian": "hr_core_news_sm",
    "danish": "da_core_news_sm",
    "dutch": "nl_core_news_sm",
    "finnish": "fi_core_news_sm",
    "greek": "el_core_news_sm",
    "japanese": "ja_core_news_sm",
    "korean": "ko_core_news_sm",
    "lithuanian": "lt_core_news_sm",
    "macedonian": "mk_core_news_sm",
    "polish": "pl_core_news_sm",
    "portuguese": "pt_core_news_sm",
    "romanian": "ro_core_news_sm",
    "russian": "ru_core_news_sm",
    "spanish": "es_core_news_sm",
    "swedish": "sv_core_news_sm",
    "ukrainian": "uk_core_news_sm"
}


ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Misses",
    "No.": "Number",
    "Dr.": "Doctor",
    "Ms.": "Miss",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Ln.": "Lane",
    "Rd.": "Road",
    "a.m.": "before noon",
    "p.m.": "after noon",
    "ft.": "feet",
    "hr.": "hour",
    "min.": "minute",
    "sq.": "square",
    "St.": "street",
    "Asst.": "assistant",
    "Corp.": "corporation"
}

ISWORD = re.compile(r'.*\w.*')
import sys, os
def get_base_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return base_path
def split_audio_from_video(video_file):
    try:
        print("Extracting audio track")
        video = VideoFileClip(video_file, audio=True)
        audio = video.audio
        audio_file = os.path.splitext(video_file)[0] + ".wav"
        audio.write_audiofile(audio_file, logger=None)
        return audio_file
    except Exception as e:
        print(f"Error extracting audio from video: {e}")
        config.logger.info(f"Error extracting audio from video: {e}")
        return None

import numpy as np
def get_silent_data(sr, dur):
    zero_wav = np.zeros(
        int(sr * dur),
        dtype=np.float32
    )
    return zero_wav
def transcribe_fun(audio):
    from funasr import AutoModel
    import numpy as np
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      #punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      )
    #print("raw audio:", audio)
    #audio = list(audio)
    #audio = np.frombuffer(audio, dtype='uint8')
    print("transcribe audio shape:", audio.shape)

    res = model.generate(input=audio,
                batch_size_s=300,
                hotword='')
    # text: 'their hand', 'timestamp': [[270,470],[470, 810]]
    return res

def transcribe_audio(audio_file, source_language):
    try:
        print("Transcribing audio track")
        model_dir = os.path.join(get_base_path(),"./models/whisper/tiny.pt")
        model = whisper.load_model(model_dir)#large
        #model = whisper.load_model("tiny", download_root="./models/whisper")  # large
        trans = model.transcribe(audio_file, language=source_language, verbose=False, word_timestamps=True)
        return trans
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


def translate_text(texts, target_language):
    from microsoft import trans
    logout = False
    target_language = 'zh-Hans'
    text = ''.join(texts)
    #print(" to translate:", text)
    try:
        results = trans(text, target_language, set_p=logout, inst=None, source_code="")
        #print("trans result:", results)
        return [results]  # [result['translatedText'] for result in results]
    except Exception as e:
        print(f"Error translating texts: {e}")
        return None

import time
def create_audio_from_text(text, target_language, role):

    audio_file = "./translated_"  + str(uuid.uuid4()) + ".wav" #
    communicate = edge_tts.Communicate(text, role, rate="+0%", volume='+0%', pitch="+2Hz")
    try:
        asyncio.run(communicate.save(audio_file))
        return audio_file
    except Exception as e:
        if os.path.isfile(audio_file):
            os.remove(audio_file)
        print("text:", text, e)
        err = str(e)
        if err.find("Invalid response status") > 0 or err.find('WinError 10054') > -1:
            time.sleep(10)
            return create_audio_from_text(
                text=text,
                target_language = target_language,
                role = role
            )
        print(f"Error creating audio from text: {e}")
        return None
        #raise Exception(f"Error creating audio from text: {e}")
def split_text_by_comma(text):
    import re
    sentences = re.split(r'[，,。！!；;？?]', text[0])
    #sentences = [s for s in re.split(r'[，,。！!；;？?]', text) if s]
    return sentences
def get_silent_parts(vad):
    si_parts = []
    #print("silent parts:", vad, len(vad))
    for i in range(len(vad)):
        if i == len(vad) -1:
            break
        if vad[i][1] != vad[i+1][0]:
            si_parts.append([vad[i][1], vad[i+1][0]])
    return si_parts
def get_sentence_silents(sent_start, sent_end, all_silents):
    si_parts = []
    for it in all_silents:
        if it[0] >= sent_start and it[1] <= sent_end:
            si_parts.append(it)
            
    return si_parts

def check_word_range(start, end, vad):
    new_start = start
    new_end = end
    if None == vad:
        return start, end
    for it in vad:
        if start < it[0] and (end > it[0] + 0.1) and (end + 0.1) <= it[1]:# [9.7, 10.68] [10.675, 13.125]
            print("start check:", start, it[0], it[1])
            new_start = it[0] #check start
        if end > it[1] and start < it[1] and start >= (it[0]):#[9.7, 10.68], [6.77, 9.85]
            print("end check:", end, it[0], it[1])
            new_end = it[1] #check end
        if start < it[0] and end > it[1]:
            print("error range:", start,end, it[0], it[1])

    return new_start, new_end

def select_vad(seg_start,seg_end, vad):
    vad_my = []
    for it in vad:
        if it[0] <= seg_start and it[1] >= seg_start:
            vad_my.append(it)
        if it[0] >= seg_start and it[0] <= seg_end:
            vad_my.append(it)
        if it[0] >= seg_start and it[1] <= seg_end:
            vad_my.append(it)
    return vad_my
import jieba.posseg as pseg
import jieba

def module_sub_sent(words, flags, sent_start, sent_end):
    speed_org = len(words) / (sent_end - sent_start)
    #eval voice len, remove some words if speed too slow

def remove_words(txt, target_len):
    words = pseg.cut(txt)
    comma = "，,。.！!；;？?"
    vir_words = ['uj', 'p','ul']
    remove_len = len(txt) - target_len
    
    removed_len = 0

    for word, flag in words:
        if word in comma:
            continue
        if removed_len >= remove_len:
            break
        if flag in vir_words:
            len_b = len(txt)
            print("remove word:", word)
            txt = txt.replace(word, "")
            removed_len += len_b - len(txt)
    return txt

def change_dynamic(audio, sample_rate):
    audio_normalized = audio / np.max(np.abs(audio))

    threshold = 0.1
    compression_ratio = 0.5

    window_size = int(sample_rate * 0.01)
    audio_squared = audio_normalized ** 2
    energy = np.sqrt(np.convolve(audio_squared, np.ones(window_size) / window_size, mode='same'))

    gain = np.ones_like(audio_normalized)
    for i in range(len(gain)):
        if energy[i] > threshold:
            gain[i] = 1 - compression_ratio * (energy[i] - threshold) / (1 - threshold)
        #gain[i] = gain[i] * (len(gain) - i/80) / i

    audio_compressed = audio_normalized * gain
    rescale = 1
    if max(audio_compressed)/np.max(np.abs(audio)) > 1.1:
        rescale = np.max(np.abs(audio)) * 1.1 / max(audio_compressed)
    audio_compressed *= rescale
    return audio_compressed
import pybungee
def change_speed_my(audio, sr, speed):
    outputData = []
    pitch = 0
    #print("input len:", len(audio))
    inSampleRate = sr
    outSampleRate = sr
    outputData = pybungee.process(audio, inSampleRate, outputData, outSampleRate, speed, pitch)
    outputData = change_dynamic(outputData, sr)
    #print("outlen:", len(outputData))
    return outputData
def get_valid_word_cnt(text):
    comma = "，,。.！!；;？?"
    new_txt = "".join(filter(lambda x: x not in comma, text))
    return len(new_txt), new_txt
def get_nosilence_range(audio_file):
    audio_target = AudioSegment.from_file(audio_file)
    nosi_target = silence.detect_nonsilent(audio_target, min_silence_len=260, silence_thresh=-50, seek_step=5)
    #print("nosi:", nosi_target)
    vad_target = [[it/1000.0 for it in rwo] for rwo in nosi_target] 
    
    return vad_target

def split_sentece_by_silent(sentence_txt, sent_start, sent_end, sent_silents):
    if not sent_silents:
        print("no silents")
        words = pseg.cut(sentence_txt)
        #for word, flag in words:
         #   print("word,flag", word, flag)
        return [sentence_txt],[[sent_start, sent_end]], [0]
    words = pseg.cut(sentence_txt)# default false , cut_all=False
    comma = "，,。.！!；;？?"
    sent_len = 0
    new_sent = []
    new_sub_sent_seg = [] # range after split
    # get len
    words_list = []
    flag_list = []
    for word, flag in words:
        #print("word,flag", word, flag)
        words_list.append(word)
        flag_list.append(flag)
        #if word in comma:
         #   continue
        sent_len += 1
        
    print("sentence length:", sent_len)
    si_pos = []
    for si in sent_silents:# sent_silents is the silent in the sentence
        sp = int(sent_len * (si[0] - sent_start) / (sent_end - sent_start) )
        if sp == 0 and si[0] > sent_start:
            sp = 1
        if sp == 0 or sp == sent_len:
            continue

        si_pos.append(sp)

    si_dur = [] # silence duration
    for i,p in enumerate(si_pos):
        #print("i,p:", i, p)
        if 0 == i:
            #print("to add first:", "".join(words_list[0:p]))
            new_sent.append("".join(words_list[0:p]))
            new_sub_sent_seg.append([sent_start, sent_silents[i][0]])# incorrect if len(si_pos) != len(sent_silents)
            si_dur.append(sent_silents[i][1] - sent_silents[i][0])
        else:
            new_sent.append("".join(words_list[si_pos[i - 1]: p]))
            new_sub_sent_seg.append([sent_silents[i-1][1], sent_silents[i][0]])
            si_dur.append(sent_silents[i][1] - sent_silents[i][0])
        if i == len(si_pos) - 1:
            new_sent.append("".join(words_list[si_pos[i]:]))
            new_sub_sent_seg.append([sent_silents[i][1], sent_end])
            si_dur.append(0)# last one
    return new_sent, new_sub_sent_seg, si_dur

import zhconv
def parse_transcription(transcription, vad, source_language):#for chiniese only 
    sentences = ""
    sentence_range= []

    sentence = ""
    sent_start = 0
    sent_end = 0
    words_range = []

    for segment in tqdm(transcription["segments"]):
        if segment["text"].isupper():
            continue
        
        vad_seg = select_vad(segment["start"], segment["end"], vad)
        #print("seg start end:", segment["start"], segment["end"])

        
        for i, word in enumerate(segment["words"]):
            if not ISWORD.search(word["word"]):
                continue
            word["word"] = ABBREVIATIONS.get(word["word"].strip(), word["word"])
            if word["word"].startswith("-"):
                sentence = sentence[:-1] + word["word"] #+ " "
            else:
                sentence += word["word"] #+ " " chinese no space

            word["start"], word["end"] = check_word_range(word["start"], word["end"], vad_seg)
            #print("after check word segm:", word["word"], word["start"], word["end"])
            if 1 == len(word["word"]):
                words_range.append([word["start"], word["end"]])
            if len(word["word"]) > 1:
                wl = len(word["word"])
                wstep = (word["end"] - word["start"]) / wl
                for j in range(wl):
                    words_range.append([word["start"] + j * wstep, word["start"] + (j+1) * wstep])
            if i == 0: 
                sent_start = word["start"]
            if i == len(segment["words"]) -1:
                sent_end = word["end"]
            #print("word_speed:", word_speed, word["end"], word["start"])
        print("lens:", len(sentence), len(segment["text"]), len(words_range))
        sentences +=sentence
        sentence_range.append([sent_start, sent_end])

        sentence = ""
    print("berfor:", sentences)
    sentences = zhconv.convert(sentences, 'zh-hans')#转简体
    print("words range:", len(words_range), sentences)
    return sentences, sentence_range, words_range


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms

    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms
def remove_lead_tail_silence(audio, audio_file, samplerate):
    sound = AudioSegment.from_file(audio_file)
    start_trim = detect_leading_silence(sound)#ms
    end_trim = detect_leading_silence(sound.reverse())
    audio = audio[int(start_trim*samplerate/1000): len(audio) - int(end_trim*samplerate/1000)]
    #print("Trim ms:", start_trim, end_trim)
    return audio
def test_tts_speed(target_language, target_voice):
    test_txt = "注意啦这是测试的文字" #10 charaters
    translated_audio_file = create_audio_from_text(test_txt, target_language, target_voice)

    if translated_audio_file is None:
        raise Exception("Audio creation failed")

    translated_audio = AudioSegment.from_file(translated_audio_file)
    start_trim = detect_leading_silence(translated_audio)
    end_trim = detect_leading_silence(translated_audio.reverse())
    #print("Trim ms:", start_trim, end_trim, len(translated_audio))
    translated_audio = translated_audio[start_trim: len(translated_audio) - end_trim]
    speed = len(translated_audio)/10 # 10是字的个数
    #print("Speed:", speed)
    
    os.remove(translated_audio_file)
    return speed

def translate_audio_files_fun(transcription, source_language, target_language, target_voice, audio_file):
    temp_files = []

    if spacy_models[source_language] not in spacy.util.get_installed_models():
        spacy.cli.download(spacy_models[source_language])
    nlp = spacy.load(spacy_models[source_language])
    nlp.add_pipe("syllables", after="tagger")
    merged_audio = AudioSegment.silent(duration=0)
    txt = transcription[0]['text']
    sentences = transcription[0]['text'].split()
    ts = transcription[0]['timestamp']
    sentence_starts = []
    sentence_ends = []
    sentence = ""
    sent_start = 0
    speeds = []

def save_audio_to_file(audio, filename):
    try:
        #audio.export(filename, format="wav")
        sf.write(filename, audio, 16000)
        print(f"Audio track with translation only saved to {filename}")
    except Exception as e:
        print(f"Error saving audio to file: {e}")



def replace_audio_in_video(video_file, new_audio, target_path=None):
    try:
        # Load the video
        video = VideoFileClip(video_file)

        # Save the new audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            #new_audio.export(temp_audio_file.name, format="wav")
            #print("to save:", new_audio, new_audio.dtype)
            sf.write(temp_audio_file.name, new_audio, 16000)
            #print("save over")
        #new_audio.export("duckled.wav", format="wav")

        # Load the new audio into an AudioFileClip
        try:
            new_audio_clip = AudioFileClip(temp_audio_file.name)
        except Exception as e:
            print(f"Error loading new audio into an AudioFileClip: {e}")
            return

        # Check audio duration
        if new_audio_clip.duration < video.duration:
            print("Warning: new audio shorter than video, with remaining video no sound.")
        elif new_audio_clip.duration > video.duration:
            print("Warning: new audio longer than video, extra cut off.")
            new_audio_clip = new_audio_clip.subclip(0, video.duration)

        # Set the audio of the video to the new audio
        video = video.set_audio(new_audio_clip)

        # Write the result to a new video file
        output_filename = os.path.splitext(video_file)[0] + "_translated.mp4"
        if target_path != None:
            output_filename = os.path.join(target_path, os.path.splitext(os.path.basename(video_file))[0] + "_translated.mp4")
        try:
            video.write_videofile(output_filename, audio_codec='aac', write_logfile=False, logger= None)
        except Exception as e:
            print(f"Error writing the new video file: {e}")
            return
        new_audio_clip.close()
        print(f"Translated video saved as {output_filename}")

    except Exception as e:
        print(f"Error replacing audio in video: {e}")
    finally:
        # Remove the temporary audio file
        if os.path.isfile(temp_audio_file.name):
            os.remove(temp_audio_file.name)

def generate_endp(audio):
    from funasr import AutoModel

    model = AutoModel(model="fsmn-vad")
    res = model.generate(input=audio)  # default 16k
    #print(res)
    val = res[0]['value']
    return [[it/1000.0 for it in rwo] for rwo in val] #[ [],[]]

def fun_para(audio):
    from funasr import AutoModel
    model_dir = os.path.join(get_base_path(), "models/paraformer/") #"paraformer-zh"
    model = AutoModel(model= model_dir, model_revision="v2.0.4", vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      #punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                      # spk_model="cam++", spk_model_revision="v2.0.2",
                      )
    print("model:", model)
    print("audio:", audio.shape)

    res = model.generate(input=audio,
                batch_size_s=300,
                hotword='')

    print("len:", len(res[0]["text"].split()), len(res[0]["timestamp"]))# 'timestamp': [[50, 230], [230, 390], [390, 570], unit ms , txt 那 第 三 点 呢
    print(res)
    txt = "".join(res[0]["text"].split())
    ts = res[0]["timestamp"]
    ts = [[it/1000.0 for it in rwo] for rwo in ts]
    return txt, ts

def audio_pos(ms, sr):
    return int(ms *sr / 1000)

def parse_transc_whisper(transcription, vad):
    sentences = []
    sentence_starts = []
    sentence_ends = []
    sentence = ""
    sent_source = ""
    sent_start = 0
    for segment in tqdm(transcription["segments"]):
        if segment["text"].isupper():
            continue
        vad_seg = select_vad(segment["start"], segment["end"], vad)
        print("seg start end:", segment["start"], segment["end"])
        for i, word in enumerate(segment["words"]):
            if not ISWORD.search(word["word"]):
                continue
            word["word"] = ABBREVIATIONS.get(word["word"].strip(), word["word"])
            if word["word"].startswith("-"):
                sentence = sentence[:-1] + word["word"] + " "
            else:
                sentence += word["word"] + " "
            
            word_syllables = len(word["word"])#sum(token._.syllables_count for token in nlp(word["word"]) if token._.syllables_count)
            segment_syllables = len(segment["text"])#sum(token._.syllables_count for token in nlp(segment["text"]) if token._.syllables_count)

            word_speed = word_syllables / (word["end"] - word["start"])

            if sent_start == 0:#i == 0 or 
                sent_start = word["start"]
                word_speed = word_syllables / (word["end"] - word["start"])

            if i == len(segment["words"]) - 1:  # last word in segment
                word_speed = word_syllables / (word["end"] - word["start"])
                segment_speed = segment_syllables / (segment["end"] - segment["start"])
                if word_speed < 1.0 or segment_speed < 2.0:# too slow, then add segment
                    print("too slow speed")
                    word["word"] += "."
                #print("all speeds:", segment["words"], speeds)

            if word["word"].endswith(".") or word["word"].endswith(","):
                #print("sentence:", sentence, sent_start, word["end"])
                sentences.append(sentence)
                sent_source += sentence
                sentence_starts.append(sent_start)
                sentence_ends.append(word["end"])
                sent_start = 0
                sentence = ""    
    return sentences,sent_source, sentence_starts, sentence_ends


