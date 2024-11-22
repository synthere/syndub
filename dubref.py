import argparse
import os
import uuid
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip, AudioFileClip
import config
import whisper
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

import jieba.posseg as pseg
import jieba

from util import *

def find_match_pos(source_txt, txt_to_match):
    len_source = len(source_txt)
    len_match = len(txt_to_match)
    match_dur = 5
    match_ind = -1
    match_source_pos = -1
    for i in range(len_match - match_dur):
        match_ind = source_txt.find(txt_to_match[i:i+match_dur])
        if match_ind != -1:
            #print("match found:", txt_to_match[i:i+match_dur], match_ind, i)
            match_source_pos = i
            break
    return match_ind, match_source_pos

def audio_pos(ms, sr):
    return int(ms *sr / 1000)

class VideoDubRef():
    def __init__(self):
        self.translate_percent = 0.0
        self.split_cost = 1
        self.prepare_cost = 5
        self.translate_cost = 20
        self.adjust_tts_cost = 59
        self.replace_cost = 15

    def update_progress(self, increse_percent):
        #print("increese:", increse_percent)
        self.translate_percent += increse_percent
        if self.translate_percent > 100:
            self.translate_percent = 100

    def trans(self, video_file, source_lang, target_lang, target_voice_file, target_path):
        start_time = time.time()
        audio_file = extract_audio_from_video(video_file)
        if audio_file is None:
            config.logger.info('no audio file in video')
            return

        self.update_progress(self.split_cost)

        merged_audio, duck, txt_source, txt_fin = self.merge_audio_files(source_lang, target_lang, target_voice_file, audio_file)

        if merged_audio is None:
            config.logger.info('result in no audio')
            return
        print("to replacess")
        # self.update_progress(85)

        replace_audio_in_video(video_file, merged_audio, target_path)

        basn = os.path.splitext(os.path.basename(video_file))[0]
        output_filename = os.path.join(target_path, basn + ".wav")
        save_audio_to_file(merged_audio, output_filename)

        with open(os.path.join(target_path, basn + '_origin.txt'), 'w', encoding='utf-8') as file:
            file.write(txt_source)
        with open(os.path.join(target_path, basn + '_translated.txt'), 'w', encoding='utf-8') as file:
            file.write(txt_fin)
        # self.update_progress(100)

    def merge_audio_files(self, source_language, target_language, target_voice_file, audio_file):
        temp_files = []
        try:
            vad = get_nosilence_range(audio_file)
            transcription = transcribe_audio(audio_file, source_language)

            target_language = "zh-hans"
            target_audio, sr_target = librosa.load(target_voice_file, sr=16000)
            comma = "，,。.！!；;？?"

            target_sentences, target_seg = fun_para(target_audio)

            ducked_audio = AudioSegment.from_wav(audio_file)  # can be indexed by time
            merged_audio = []  # AudioSegment.silent(duration=0)

            #print("Trans")
            silent_parts = get_silent_parts(vad)
            print("silent parts:", silent_parts)

            txt_final = ""  # used to write, add pause and comma
            self.update_progress(self.prepare_cost)
            sentences, sent_source, sentence_starts, sentence_ends = parse_transc_whisper(transcription, vad)
            # translate sentences in chunks of 128
            print("Translating sentences, len:", len(sentences))
            translated_texts = []
            target_pos = 0  # target_audio_word position
            source_pos = 0
            for i in tqdm(range(0, len(sentences))):  # , 128
                chunk = sentences[i]  # sentences[i:i + 128]# 128 sentences not words
                #print("chunk to:", i, chunk)
                translated_chunk = translate_text(chunk, target_language)
                if translated_chunk is None:
                    raise Exception("Translation failed")
                translated_texts.extend(translated_chunk)
                pg = int(self.translate_cost / len(sentences))
                #print("progress:", pg)
                self.update_progress(pg)

            print("Creating translated audio track:", translated_texts)

            for i, translated_text in enumerate(tqdm(translated_texts)):
                # get silents in senctence
                si_sentence = get_sentence_silents(sentence_starts[i], sentence_ends[i], silent_parts)
                target_word_max = ((sentence_ends[i] - sentence_starts[i]) * 1000) * 1.2
                target_word_min = ((sentence_ends[i] - sentence_starts[i]) * 1000) * 0.8

                translated_text = remove_words(translated_text, target_word_max)

                splited_txt, seg_ranges, si_durs = split_sentece_by_silent(translated_text, sentence_starts[i],
                                                                           sentence_ends[i], si_sentence)

                if 0 == i:
                    padding_duration = sentence_starts[i]

                    if 0 < padding_duration:
                        padding = get_silent_data(16000, padding_duration)
                        merged_audio = np.concatenate((merged_audio, padding))

                for txt, seg_range, si_dur in zip(splited_txt, seg_ranges, si_durs):
                    #print("txt:", txt, len(txt), seg_range, si_dur, target_pos)
                    vali_len, new_txt = get_valid_word_cnt(txt)
                    # alignment
                    aligment_offset = 0
                    if vali_len < 1:
                        continue

                    if vali_len >= 5:
                        source_in_rel, target_in = find_match_pos(new_txt, target_sentences)
                        if target_in != -1:
                            source_in_abs = source_in_rel + source_pos
                            if source_in_abs != target_in:
                                aligment_offset = target_in - source_in_abs


                    # get audio from target_audio

                    vali_len_off = vali_len + aligment_offset - 1
                    tstart = target_seg[target_pos][0]

                    tend = target_seg[target_pos + vali_len_off][1]
                    translated_audio = target_audio[int(tstart * 16000):int(tend * 16000)]
                    rate = sr_target
                    target_pos += vali_len_off + 1
                    source_pos += vali_len
                    # change speed to adapt length

                    speed = len(translated_audio) / ((seg_range[1] - seg_range[0]) * 1000 * 16)
                    speed = round(speed, 2)

                    if speed < 0.8:
                        speed = 0.8
                    if speed > 1.2:
                        speed = 1.2

                    speed_audio = change_speed_my(translated_audio, rate, speed)
                    merged_audio = np.concatenate((merged_audio, speed_audio))
                    padding_duration = si_dur - (len(speed_audio) / rate - (seg_range[1] - seg_range[0]))

                    txt_final += txt
                    if 0 < padding_duration:
                        padding = get_silent_data(rate, padding_duration)
                        merged_audio = np.concatenate((merged_audio, padding))

                        if txt[-1] not in comma:
                            txt_final += "，"
                    pg = self.adjust_tts_cost / (len(translated_texts) * len(splited_txt))
                    self.update_progress(int(pg))

                if i != (len(sentence_starts) - 1):
                    padding_duration = sentence_starts[i + 1] - sentence_ends[i]
                    if 0 < padding_duration:
                        padding = get_silent_data(rate, padding_duration)
                        merged_audio = np.concatenate((merged_audio, padding))

                # merged_audio = translated_audio
                print("finis:", i)
            merged_audio = np.array(merged_audio)
            return merged_audio, ducked_audio, sent_source, txt_final
        except Exception as e:
            print(f"Error merging audio files: {e}")
            return None
        finally:
            # cleanup: remove all temporary files
            for file in temp_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing temporary file {file}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the source video file', required=True)
    parser.add_argument('--voice', type=str, default="",
                        help=f'Target dubbing voice file name')

    parser.add_argument('--source_language', type=str, help=f'Source language, e.g. english.', default="english")
    args = parser.parse_args()

    audio_file = extract_audio_from_video(args.input)#args.input #
    return
    if audio_file is None:
        return
    audio, sr = librosa.load(audio_file, sr=16000)
    segments = generate_endp(audio)
    print("vad segments:", segments)
    
    audio_seg = AudioSegment.from_file(audio_file)
    nosi = silence.detect_nonsilent(audio_seg, min_silence_len=260, silence_thresh=-50, seek_step=5)
    #print("nosi:", nosi)
    segments = [[it/1000.0 for it in rwo] for rwo in nosi]
    
    transcription = transcribe_audio(audio_file, args.source_language)
    print("voicefile:", args.voice)
    #transc_target = transcribe_audio(args.voice, 'chinese')# mandarin fanti
    print("transc:", transcription)

    dub = VideoDubRef()
    target_language = 'zh-Hans'

    merged_audio, duck, txt_source, txt_fin = dub.merge_audio_files(args.source_language.lower(), target_language, args.voice, audio_file)
    if merged_audio is None:
        return
    print("to replacess")
    #replace_audio_in_video(args.input, merged_audio)
    # Save the audio file with the same name as the video file but with a ".wav" extension
    output_filename = os.path.splitext(args.input)[0] + "_resl.wav"
    save_audio_to_file(merged_audio, output_filename)

    with open('transc.txt', 'w', encoding='utf-8') as file:
        file.write(txt_source)
        
    with open('translated.txt', 'w', encoding='utf-8') as file:
        file.write(txt_fin)

if __name__ == "__main__":
    main()
