import torch
import yt_dlp as youtube_dl
from pydub import AudioSegment
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

def download_youtube_audio(youtube_url, output_format='wav'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': output_format,
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'outtmpl': '%(id)s.%(ext)s',
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def transcribe_audio(file_path, batch_size=48000):
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
    model.to("cuda")
    
    audio = AudioSegment.from_file(file_path)
    audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        audio_array = audio_array.reshape((-1, audio.channels)).mean(axis=1)
    audio_array /= 32768.0

    input_values = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    total_samples = input_values.shape[1]
    transcription = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = start_idx + batch_size
        batch = input_values[:, start_idx:end_idx].to("cuda")

        with torch.no_grad():
            logits = model(batch).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription_part = processor.batch_decode(predicted_ids)
        transcription.append(transcription_part[0])

    full_transcription = ' '.join(transcription)
    return full_transcription

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=VmPuh7wUEfY"
    download_youtube_audio(url)
    video_id = url.split("=")[-1]
    transcription = transcribe_audio(f"{video_id}.wav")
    
    # Save transcription to a file
    with open(f"{video_id}_transcription.txt", "w", encoding='utf-8') as text_file:
        text_file.write("Transcription:\n")
        text_file.write(transcription)
    
    print(f"Transcription saved to {video_id}_transcription.txt")
