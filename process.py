import argparse
import os
import uuid
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
import spacy

from tqdm import tqdm
import tempfile
import re
from elevenlabs import generate, Voice, set_api_key

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


class Process:
    def __init__(self, inputfile, output):
        self.input_file = inputfile
        self.output_file = output
    def extract_audio_from_video(self, video_file):
        try:
            print("Extracting audio track")
            video = VideoFileClip(video_file)
            audio = video.audio
            audio_file = os.path.splitext(video_file)[0] + ".wav"
            audio.write_audiofile(audio_file)
            return audio_file
        except Exception as e:
            print(f"Error extracting audio from video: {e}")
            return None


    def transcribe_audio(self, audio_file, source_language):
        try:
            print("Transcribing audio track")
            model = whisper.load_model("large")
            trans = model.transcribe(audio_file, language=source_language, verbose=False, word_timestamps=True)
            return trans
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None


    def translate_text(self, texts, target_language):
        from microsoft import trans
        logout = True
        try:
            results = trans(texts, target_language, set_p=logout, inst=None, source_code="")
            return results #[result['translatedText'] for result in results]
        except Exception as e:
            print(f"Error translating texts: {e}")
            return None


    def create_audio_from_text(self, text, target_language, target_voice):
        audio_file = "translated_" + str(uuid.uuid4()) + ".wav"
        try:
            set_api_key("sk_224a118e9b4418b835f7a356a8ab8c4b1bf3befab4051ac5")
            audio = generate(
                text=text,
                voice="zrHiDhphv9ZnVXBqCLjz",
                model="eleven_multilingual_v2"
            )
            with open(audio_file, "wb") as out:
                out.write(audio)
            return audio_file
        except Exception as e:
            if os.path.isfile(audio_file):
                os.remove(audio_file)
            raise Exception(f"Error creating audio from text: {e}")

    def merge_audio_files(self, transcription, source_language, target_language, target_voice, audio_file):
        temp_files = []
        try:
            ducked_audio = AudioSegment.from_wav(audio_file)
            if spacy_models[source_language] not in spacy.util.get_installed_models():
                spacy.cli.download(spacy_models[source_language])
            nlp = spacy.load(spacy_models[source_language])
            nlp.add_pipe("syllables", after="tagger")
            merged_audio = AudioSegment.silent(duration=0)
            sentences = []
            sentence_starts = []
            sentence_ends = []
            sentence = ""
            sent_start = 0
            print("Composing sentences")
            for segment in tqdm(transcription["segments"]):
                if segment["text"].isupper():
                    continue
                for i, word in enumerate(segment["words"]):
                    if not ISWORD.search(word["word"]):
                        continue
                    word["word"] = ABBREVIATIONS.get(word["word"].strip(), word["word"])
                    if word["word"].startswith("-"):
                        sentence = sentence[:-1] + word["word"] + " "
                    else:
                        sentence += word["word"] + " "
                    # this is a trick to compensate the absense of VAD in Whisper
                    word_syllables = sum(token._.syllables_count for token in nlp(word["word"]) if token._.syllables_count)
                    segment_syllables = sum(token._.syllables_count for token in nlp(segment["text"]) if token._.syllables_count)
                    if i == 0 or sent_start == 0:
                        word_speed = word_syllables / (word["end"] - word["start"])
                        if word_speed < 3:
                            sent_start = word["end"] - word_syllables / 3
                        else:
                            sent_start = word["start"]
                    if i == len(segment["words"]) - 1:  # last word in segment
                        word_speed = word_syllables / (word["end"] - word["start"])
                        segment_speed = segment_syllables / (segment["end"] - segment["start"])
                        if word_speed < 1.0 or segment_speed < 2.0:
                            word["word"] += "."

                    if word["word"].endswith("."):
                        sentences.append(sentence)
                        sentence_starts.append(sent_start)
                        sentence_ends.append(word["end"])
                        sent_start = 0
                        sentence = ""
            # translate sentences in chunks of 128
            print("Translating sentences")
            translated_texts = []
            for i in tqdm(range(0, len(sentences), 128)):
                chunk = sentences[i:i + 128]
                translated_chunk = self.translate_text(chunk, target_language)
                if translated_chunk is None:
                    raise Exception("Translation failed")
                translated_texts.extend(translated_chunk)
            print("Creating translated audio track")
            prev_end_time = 0
            for i, translated_text in enumerate(tqdm(translated_texts)):
                translated_audio_file = self.create_audio_from_text(translated_text, target_language, target_voice)
                if translated_audio_file is None:
                    raise Exception("Audio creation failed")
                temp_files.append(translated_audio_file)
                translated_audio = AudioSegment.from_wav(translated_audio_file)

                # Apply "ducking" effect: reduce volume of original audio during translated sentence
                start_time = int(sentence_starts[i] * 1000)
                end_time = start_time + len(translated_audio)
                next_start_time = int(sentence_starts[i+1] * 1000) if i < len(translated_texts) - 1 else len(ducked_audio)
                ducked_segment = ducked_audio[start_time:end_time].apply_gain(-10)  # adjust volume reduction as needed

                fade_out_duration = min(500, max(1, start_time - prev_end_time))
                fade_in_duration = min(500, max(1, next_start_time  - end_time))
                prev_end_time = end_time
                # Apply fade in effect to the end of the audio before the ducked segment
                if start_time == 0:
                    ducked_audio = ducked_segment +  ducked_audio[end_time:].fade_in(fade_in_duration)
                elif end_time == len(ducked_audio):
                    ducked_audio = ducked_audio[:start_time].fade_out(fade_out_duration) + ducked_segment
                else:
                    ducked_audio = ducked_audio[:start_time].fade_out(fade_out_duration) \
                                   + ducked_segment +  ducked_audio[end_time:].fade_in(fade_in_duration)

                # Overlay the translated audio on top of the original audio
                ducked_audio = ducked_audio.overlay(translated_audio, position=start_time)

                original_duration = int(sentence_ends[i] * 1000)
                new_duration = len(translated_audio) + len(merged_audio)
                padding_duration = max(0, original_duration - new_duration)
                padding = AudioSegment.silent(duration=padding_duration)
                merged_audio += padding + translated_audio
            return merged_audio, ducked_audio
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


    def save_audio_to_file(self, audio, filename):
        try:
            audio.export(filename, format="wav")
            print(f"Audio track with translation only saved to {filename}")
        except Exception as e:
            print(f"Error saving audio to file: {e}")



    def replace_audio_in_video(self, video_file, new_audio):
        try:
            # Load the video
            video = VideoFileClip(video_file)

            # Save the new audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                new_audio.export(temp_audio_file.name, format="wav")
            new_audio.export("duckled.wav", format="wav")

            # Load the new audio into an AudioFileClip
            try:
                new_audio_clip = AudioFileClip(temp_audio_file.name)
            except Exception as e:
                print(f"Error loading new audio into an AudioFileClip: {e}")
                return

            # Check if the audio is compatible with the video
            if new_audio_clip.duration < video.duration:
                print("Warning: The new audio is shorter than the video. The remaining video will have no sound.")
            elif new_audio_clip.duration > video.duration:
                print("Warning: The new audio is longer than the video. The extra audio will be cut off.")
                new_audio_clip = new_audio_clip.subclip(0, video.duration)

            # Set the audio of the video to the new audio
            video = video.set_audio(new_audio_clip)

            # Write the result to a new video file
            output_filename = os.path.splitext(video_file)[0] + "_translated.mp4"
            try:
                video.write_videofile(output_filename, audio_codec='aac')
            except Exception as e:
                print(f"Error writing the new video file: {e}")
                return

            print(f"Translated video saved as {output_filename}")

        except Exception as e:
            print(f"Error replacing audio in video: {e}")
        finally:
            # Remove the temporary audio file
            if os.path.isfile(temp_audio_file.name):
                os.remove(temp_audio_file.name)


def main():
    parser = argparse.ArgumentParser()

if __name__ == "__main__":
    main()
