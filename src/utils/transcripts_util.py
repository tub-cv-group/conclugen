from typing import List, Union
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from scipy.io import wavfile
import torch


model, processor = None, None


def _extract_transcript_from_wave_file(inputs) -> str:
    global model
    if model is None:
        model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")

    features = inputs["input_features"]
    attention_masks = inputs["attention_mask"]
    if torch.cuda.is_available():
        model = model.cuda()
        features = features.cuda()
        attention_masks = attention_masks.cuda()

    generated_ids = model.generate(features, attention_mask=attention_masks)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    transcription = transcription
    return transcription


def extract_transcripts_from_wave_files(audio_file_or_files: Union[str, List[str]], out_file_or_files: None) -> str:
    global processor
    if processor is None:
        processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
    if isinstance(audio_file_or_files, list):
        audio_data = []
        max_length = -1
        for audio_file in audio_file_or_files:
            sampling_rate, data = wavfile.read(audio_file)
            audio_data.append(data)
            if max_length < data.shape[0] or max_length == -1:
                max_length = data.shape[0]
        inputs = processor(
            audio_data,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            max_length=max_length,
            padding='longest')
    else:
        sampling_rate, audio_data = wavfile.read(audio_file_or_files)
        audio_data = torch.tensor(audio_data)
        inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
    transcription = _extract_transcript_from_wave_file(inputs)
    if out_file_or_files is not None:
        if isinstance(out_file_or_files, list):
            for i, out_file in enumerate(out_file_or_files):
                with open(out_file, 'w+') as f:
                    f.write(transcription[i])
        else:
            with open(out_file_or_files, 'w+') as f:
                f.write(transcription[0])
    if isinstance(audio_file_or_files, list):
        return transcription
    else:
        return transcription[0]
