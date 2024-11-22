# -*- coding: utf-8 -*-

import utili
import librosa
import config
import numpy as np
import os, time
from pydub import AudioSegment, silence
from tqdm import tqdm
from diarization import speaker_diarization
class VideoDubbing():
    def __init__(self):
        self.translate_percent = 0.0
        self.split_cost = 1
        self.prepare_cost = 5
        self.translate_cost = 20
        self.adjust_tts_cost = 59
        self.replace_cost = 15
    
    def update_progress(self, increse_percent):
        self.translate_percent += increse_percent
        if self.translate_percent > 100:
            self.translate_percent = 100

    def trans(self, video_file, source_lang, target_lang, role, target_path, speaker_num):
        start_time = time.time()
        audio_file = utili.split_audio_from_video(video_file)
        if audio_file is None:
            config.logger.info('no audio file in video')
            return

        self.update_progress(self.split_cost)

        merged_audio, duck, txt_source, txt_fin = self.translate_audio_files(source_lang, target_lang, role, audio_file, speaker_num)

        if merged_audio is None:
            config.logger.info('result in no audio')
            return
        print("to replacess")

        utili.replace_audio_in_video(video_file, merged_audio, target_path)

        basn = os.path.splitext(os.path.basename(video_file))[0]
        output_filename = os.path.join(target_path, basn + ".wav")
        utili.save_audio_to_file(merged_audio, output_filename)

        with open(os.path.join(target_path,  basn + '_origin.txt'), 'w', encoding='utf-8') as file:
            file.write(txt_source)    
        with open(os.path.join(target_path,  basn + '_translated.txt'), 'w', encoding='utf-8') as file:
            file.write(txt_fin)
        #self.update_progress(100)

    @staticmethod
    def get_role(roles, seg_rang, speak_range):
        very_role = roles[0]
        role_num = len(roles)
        tolerance = 0.2

        for rang in speak_range:
            if ((seg_rang[0] + tolerance) >= rang[1]
                    and seg_rang[1] <= (rang[2] + tolerance)
                    and rang[0] < role_num):
                very_role = roles[rang[0]]
                print(f"seg rang:{seg_rang}, speak num:{rang[0]}")
                break
        return very_role

    def translate_audio_files(self, source_language, target_language, role, audio_file, speaker_num = 1):
        """
        role []
        """
        temp_files = []
        try:
            vad = utili.get_nosilence_range(audio_file)
            print("vad:", vad)
            speaker_range = None
            if speaker_num > 1:
                speaker_range = speaker_diarization.diarize(audio_file, speaker_num)
                speak_vad = [[i[1], i[2]] for i in speaker_range]
                vad = speak_vad
                print("speak rang:", speaker_range, speak_vad)
            transcription = utili.transcribe_audio(audio_file, source_language)
            print("transc:", transcription)

            ducked_audio = AudioSegment.from_wav(audio_file)# can be indexed by time

            merged_audio = []#AudioSegment.silent(duration=0)
            sentences = []
            sentence_starts = []
            sentence_ends = []
            sentence = ""
            sent_start = 0
            sent_source = ""
            speeds = []
            silent_parts = utili.get_silent_parts(vad)

            print("Composing sentences silence:", silent_parts)
            for segment in tqdm(transcription["segments"]):
                if segment["text"].isupper():
                    continue
                vad_seg = utili.select_vad(segment["start"], segment["end"], vad)
                #print("seg start end:", segment["start"], segment["end"])
                for i, word in enumerate(segment["words"]):
                    if not utili.ISWORD.search(word["word"]):
                        continue
                    word["word"] = utili.ABBREVIATIONS.get(word["word"].strip(), word["word"])
                    if word["word"].startswith("-"):
                        sentence = sentence[:-1] + word["word"] + " "
                    else:
                        sentence += word["word"] + " "

                    word["start"], word["end"] = utili.check_word_range(word["start"], word["end"], vad_seg)
   
                    word_syllables = len(word["word"])#sum(token._.syllables_count for token in nlp(word["word"]) if token._.syllables_count)
                    segment_syllables = len(segment["text"])#sum(token._.syllables_count for token in nlp(segment["text"]) if token._.syllables_count)

                    word_speed = word_syllables / (word["end"] - word["start"])

                    if sent_start == 0:#i == 0 or 
                        sent_start = word["start"]
                        word_speed = word_syllables / (word["end"] - word["start"])
                        #if word_speed < 3:
                        speeds.append(word_speed)

                    if i == len(segment["words"]) - 1:  # last word in segment
                        word_speed = word_syllables / (word["end"] - word["start"])
                        segment_speed = segment_syllables / (segment["end"] - segment["start"])
                        if word_speed < 1.0 or segment_speed < 2.0:# too slow, then add segment
                            word["word"] += "."

                    if word["word"].endswith(".") or word["word"].endswith(","):
                        sentences.append(sentence)
                        sentence_starts.append(sent_start)
                        sentence_ends.append(word["end"])
                        sent_start = 0
                        sent_source += sentence
                        sentence = ""
            
            self.update_progress(self.prepare_cost)

            txt_final = ""
            # translate sentences in chunks of 128
            print("Translating sentences, len:", len(sentences))
            translated_texts = []
            for i in tqdm(range(0, len(sentences), 1)):#, 128
                chunk = sentences[i:i + 1]#sentences[i:i + 128]# 128 sentences not words
                #print("chunk to:", i, chunk)
                translated_chunk = utili.translate_text(chunk, target_language)
                if translated_chunk is None:
                    raise Exception("Translation failed")
                #print("chunk:", translated_chunk)
                translated_texts.extend(translated_chunk)
                pg = int(self.translate_cost / len(sentences))
                #print("progress:", pg)
                self.update_progress(pg)

            print("Creating translated audio track:", translated_texts)
            prev_end_time = 0
            # test the tts speed
            print("role:", role, len(translated_texts))
            tts_speed = utili.test_tts_speed(target_language, role[0])
            for i, translated_text in enumerate(tqdm(translated_texts)):
                #get silents in senctence
                si_sentence = utili.get_sentence_silents(sentence_starts[i], sentence_ends[i], silent_parts)
                
                eval_duration = tts_speed * len(translated_text)
                
                target_speed = eval_duration/((sentence_ends[i] - sentence_starts[i]) * 1000)
      
                target_word_max = ((sentence_ends[i] - sentence_starts[i]) * 1000) * 1.2 / tts_speed
                target_word_min = ((sentence_ends[i] - sentence_starts[i]) * 1000) * 0.8 / tts_speed
 
                translated_text = utili.remove_words(translated_text, target_word_max)

                splited_txt, seg_ranges, si_durs = utili.split_sentece_by_silent(translated_text, sentence_starts[i], sentence_ends[i], si_sentence)

                if 0 == i:
                    padding_duration = sentence_starts[i]
                    #print("begin sent padding duration:", padding_duration)
                    if 0 < padding_duration:
                        padding = utili.get_silent_data(16000, padding_duration)
                        merged_audio = np.concatenate((merged_audio,padding))
                # sentece_start = seg_ranges[0]
                for j, (txt, seg_range, si_dur) in enumerate(zip(splited_txt,seg_ranges, si_durs)):
                    very_role = role[0]
                    if speaker_range is not None:
                        very_role = self.get_role(role, seg_range, speaker_range)
                        print(f"roles:{role}, very_role:{very_role}, segrang:{seg_range}, txt:{txt}")
                    translated_audio_file = utili.create_audio_from_text(txt, target_language, very_role)

                    if translated_audio_file is None:
                        raise Exception("Audio creation failed")
                    temp_files.append(translated_audio_file)
                    translated_audio, rate = librosa.load(translated_audio_file, sr=16000)
                    translated_audio = utili.remove_lead_tail_silence(translated_audio, translated_audio_file, rate)

                    speed = len(translated_audio)/ ((seg_range[1] - seg_range[0])*1000 * 16)
                    speed = round(speed, 2)
                    print("tchangedd speed:",speed)
                    if speed < 0.85:
                        speed = 0.85
                    if speed > 1.15:
                        speed = 1.15
                    
                    speed_audio = utili.change_speed_my(translated_audio, rate, speed)
                    merged_audio = np.concatenate((merged_audio,speed_audio))
                    padding_duration = si_dur - (len(speed_audio)/rate - (seg_range[1] - seg_range[0]))

                    txt_final += txt
                    if 0 < padding_duration:
                        padding = utili.get_silent_data(rate, padding_duration)
                        merged_audio = np.concatenate((merged_audio,padding))

                    pg = self.adjust_tts_cost / (len(translated_texts) * len(splited_txt))
                    self.update_progress(int(pg))

                if i != (len(sentence_starts) -1):
                    padding_duration = sentence_starts[i + 1] - sentence_ends[i]
                    if 0 < padding_duration:
                        padding = utili.get_silent_data(rate, padding_duration)
                        merged_audio = np.concatenate((merged_audio,padding))

                print("finis:", i)

            merged_audio = np.array(merged_audio)
            return merged_audio, ducked_audio, sent_source, txt_final
        except Exception as e:
            print(f"Error merging audio files: {e}")
            config.logger.info(f"Error merg audio files: {e}")
            return None
        finally:
            # cleanup: remove all temporary files
            for file in temp_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"Error removing temporary file {file}: {e}")

  
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input video file', required=True)
    parser.add_argument('--voice', type=str, default="es-US-Neural2-B",
                        help=f'tts voice')
    parser.add_argument('--source_language', type=str, help=f'Source language, e.g. english', default="english")
    args = parser.parse_args()

    audio_file = utili.split_audio_from_video(args.input)
    if audio_file is None:
        return
    args.voice = 'zh-CN-XiaoxiaoNeural'
    dub = VideoDubbing()
    merged_audio, duck, txt_source, txt_fin  = dub.translate_audio_files(args.source_language.lower(), args.voice[:5], args.voice, audio_file)

    if merged_audio is None:
        return
    print("to replacess")
    utili.replace_audio_in_video(args.input, merged_audio)
    # Save the audio file with the same name as the video file but with a ".wav" extension
    output_filename = os.path.splitext(args.input)[0] + ".wav"
    print("output_filenamr:", output_filename, os.path.splitext(args.input))
    utili.save_audio_to_file(merged_audio, output_filename)

    with open('transc.txt', 'w', encoding='utf-8') as file:
        file.write(txt_source)
        
    with open('translated.txt', 'w', encoding='utf-8') as file:
        file.write(txt_fin)

if __name__ == "__main__":
    main()
