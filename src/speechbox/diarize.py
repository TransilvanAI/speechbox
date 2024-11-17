from typing import List, Optional, Union
import numpy as np
import requests
import torch
from pyannote.audio import Pipeline
from sympy.physics.units import length
from .utils import batch, seconds_to_srt_timestamp, get_device
from torchaudio import functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
from rich import print
from rich.progress import Progress
from .outputs import dialog_to_srt


HF_MODELS = {
    "tiny.en": "openai/whisper-tiny.en",
    "tiny": "openai/whisper-tiny",
    "base.en": "openai/whisper-base.en",
    "base": "openai/whisper-base",
    "small.en": "openai/whisper-small.en",
    "small": "openai/whisper-small",
    "medium.en": "openai/whisper-medium.en",
    "medium": "openai/whisper-medium",
    "large-v1": "openai/whisper-large-v1",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "turbo": "openai/whisper-large-v3-turbo"
}



class ASRDiarizationPipeline:
    def __init__(
            self,
            whisper_processor: WhisperProcessor,
            whisper_model: WhisperForConditionalGeneration,
            diarization_pipeline,
            voice_activity_pipeline,
            diarization_sampling_rate: int = 16000,
            device = 'cuda',
    ):
        self.whisper_processor = whisper_processor
        self.device = device
        self.whisper_model = whisper_model
        # self.asr_pipeline = asr_pipeline
        self.sampling_rate = self.whisper_processor.feature_extractor.sampling_rate
        self.diarization_sampling_rate = diarization_sampling_rate
        self.voice_activity_pipeline = voice_activity_pipeline
        self.diarization_pipeline = diarization_pipeline

    @classmethod
    def from_pretrained(
            cls,
            asr_model: Optional[str] = "openai/whisper-tiny",
            *,
            diarizer_model: Optional[str] = "pyannote/speaker-diarization-3.1",
            voice_activity_model: Optional[str] = "pyannote/voice-activity-detection",
            chunk_length_s: Optional[int] = 30,
            device='cuda',
            use_auth_token: Optional[Union[str, bool]] = True,
            flash: Optional[bool] = False,
            **kwargs,
    ):
        device = get_device(device)
        is_cpu = (device if isinstance(device, str) else getattr(device, 'type', None)) == 'cpu'
        dtype = torch.float32 if is_cpu or not torch.cuda.is_available() else torch.float16
        whisper_processor = WhisperProcessor.from_pretrained(asr_model, predict_timestamps=True, task="transcribe", language="en")
        whisper_model = (WhisperForConditionalGeneration.from_pretrained(asr_model,
                                                                        # torch_dtype=dtype,
                                                                        low_cpu_mem_usage=True,
                                                                        use_safetensors=True,
                                                                        use_flash_attention_2=flash)
                         .to(device))
        # if not flash:
        #     try:
        #         whisper_model = whisper_model.to_bettertransformer()
        #     except ValueError:
        #         pass

        diarization_pipeline = Pipeline.from_pretrained(diarizer_model, use_auth_token=use_auth_token)
        voice_activity_pipeline = Pipeline.from_pretrained(voice_activity_model, use_auth_token=use_auth_token)
        diarization_pipeline.to(device)
        voice_activity_pipeline.to(device)

        return cls(whisper_processor, whisper_model, diarization_pipeline, voice_activity_pipeline, device=device)

    def __call__(
            self,
            inputs: Union[np.ndarray, List[np.ndarray]],
            group_by_speaker: bool = True,
            legacy=False,
            bs=16,
            **kwargs,
    ):
        """
        Transcribe the audio sequence(s) given as inputs to text and label with speaker information. The input audio
        is first passed to the speaker diarization pipeline, which returns timestamps for 'who spoke when'. The audio
        is then passed to the ASR pipeline, which returns utterance-level transcriptions and their corresponding
        timestamps. The speaker diarizer timestamps are aligned with the ASR transcription timestamps to give
        speaker-labelled transcriptions. We cannot use the speaker diarization timestamps alone to partition the
        transcriptions, as these timestamps may straddle across transcribed utterances from the ASR output. Thus, we
        find the diarizer timestamps that are closest to the ASR timestamps and partition here.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is the filename of the audio file, the file will be read at the correct sampling rate
                      to get the waveform using *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be in the format `{"sampling_rate": int, "raw":
                      np.array}` with optionally a `"stride": (left: int, right: int)` than can ask the pipeline to
                      treat the first `left` samples and last `right` samples to be ignored in decoding (but used at
                      inference to provide more context to the model). Only use `stride` with CTC models.
            group_by_speaker (`bool`):
                Whether to group consecutive utterances by one speaker into a single segment. If False, will return
                transcriptions on a chunk-by-chunk basis.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update additional asr or diarization configuration parameters
                        - To update the asr configuration, use the prefix *asr_* for each configuration parameter.
                        - To update the diarization configuration, use the prefix *diarization_* for each configuration parameter.
                        - Added this support related to issue #25: 08/25/2023

        Return:
            A list of transcriptions. Each list item corresponds to one chunk / segment of transcription, and is a
            dictionary with the following keys:
                - **text** (`str` ) -- The recognized text.
                - **speaker** (`str`) -- The associated speaker.
                - **timestamps** (`tuple`) -- The start and end time for the chunk / segment.
        """
        kwargs_asr = {
            argument[len("asr_"):]: value for argument, value in kwargs.items() if argument.startswith("asr_")
        }

        kwargs_diarization = {
            argument[len("diarization_"):]: value for argument, value in kwargs.items() if
            argument.startswith("diarization_")
        }

        inputs, diarizer_inputs = self.preprocess(inputs)

        diarization = self.diarization_pipeline(
            {"waveform": diarizer_inputs, "sample_rate": self.sampling_rate},
            **kwargs_diarization,
        )

        if legacy:
            return self.__striped_legacy_call(inputs, diarization, group_by_speaker, kwargs_asr)

        diarization_dict = {
            'speakers': [],
            'segments': []
        }
        speakers_set = set()

        with Progress() as progress:
            diarization_pbar = progress.add_task("Diarization")

            for segment, label, speaker in diarization.itertracks(yield_label=True):
                speakers_set.add(speaker)
                diarization_dict['segments'].append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': speaker
                })
                progress.update(diarization_pbar, advance=0.1)

            diarization_dict['speakers'] = list(speakers_set)
            len_segments = len(diarization_dict["segments"])
            whisper_outputs = [None] * len_segments
            input_segments = []

            progress.update(diarization_pbar, completed=True)
            voice_activity_pbar = progress.add_task("Voice Activity Detection", total=len_segments)
            transcription_pbar = progress.add_task("Transcription", total=len_segments)

            for i, segment in enumerate(diarization_dict["segments"]):
                audio_segment_start = int(self.sampling_rate * segment['start'])
                audio_segment_end = int(self.sampling_rate * segment['end'])
                segment_audio = inputs[audio_segment_start:audio_segment_end]

                # Run voice activity detection on the segment
                audio_data = torch.tensor(np.row_stack((segment_audio, segment_audio)), dtype=torch.float32)
                output = self.voice_activity_pipeline({"waveform": audio_data, "sample_rate": self.sampling_rate})
                voice_activity_detection = output.get_timeline().support()
                progress.update(voice_activity_pbar, advance=1)
                # If there is no voice activity detected, then add a blank line
                if len(voice_activity_detection) == 0:
                    whisper_outputs.insert(i, {'text': None})
                    continue
                input_segments.append((i, segment_audio))

            progress.update(voice_activity_pbar, completed=True)

            for b in batch(input_segments, bs=bs):
                indexes, audio_batches = zip(*b)
                input_features = self.whisper_processor(
                    audio_batches,
                    return_attention_mask=True,
                    sampling_rate=self.sampling_rate,
                    return_token_timestamps=True,
                    return_tensors="pt",
                    truncation=False,
                    padding="longest"
                ).to(self.device)
                if input_features.input_features.shape[-1] < self.whisper_processor.feature_extractor.n_samples:
                    # we in-fact have short-form -> pre-process accordingly
                    input_features = self.whisper_processor(
                        audio_batches,
                        return_attention_mask=True,
                        sampling_rate=self.sampling_rate,
                        return_token_timestamps=True,
                        return_tensors="pt"
                    ).to(self.device)

                # we should add some stride here
                predicted_ids = self.whisper_model.generate(
                    **input_features,
                    return_timestamps=True,
                    task="transcribe",
                    # condition_on_prev_tokens=True,
                    # return_segments=True,
                    return_token_timestamps=True,
                    language="en")

                if "segments" not in predicted_ids:
                    out = [
                        {"tokens": predicted_ids["sequences"][i].unsqueeze(0),
                         "token_timestamps": predicted_ids["token_timestamps"][i].unsqueeze(0)}
                        for i in range(predicted_ids["sequences"].shape[0])
                    ]
                else:
                    # TODO: Fix this
                    out = [
                        {
                            "tokens": predicted_ids["sequences"][i].unsqueeze(0),
                            "token_timestamps": [
                                                    torch.cat([segment["token_timestamps"] for segment in segment_list])
                                                    for segment_list in predicted_ids["segments"][i].unsqueeze(0)
                                                ]
                        }
                        for i in range(predicted_ids["sequences"].shape[0])
                    ]

                transcription = self.whisper_processor.batch_decode(**predicted_ids,
                                                                    skip_special_tokens=True,
                                                                    decode_with_timestamps=False,
                                                                    output_offsets=True
                                                                    )
                # Need to check this
                word_timestamps = self.whisper_processor.tokenizer._decode_asr(out, return_timestamps="word", return_language=False, time_precision=0.2)

                for i, t in zip(list(indexes), transcription):
                    whisper_outputs.insert(i,{"word_timestamps": word_timestamps, **t}) # Fix this

                # I am not sure if we need some sor of alignement here using for example a WAV to vec model
                progress.update(transcription_pbar, advance=len(indexes))

            progress.update(transcription_pbar, completed=True)
            dialogs = self.generate_dialog(diarization_dict, whisper_outputs)
            return dialogs


    def generate_dialog(self, diarization_dict, whisper_outputs):
        result = []

        segments = diarization_dict['segments']
        speakers = diarization_dict['speakers']

        for i, segment in enumerate(segments):
            segment_text = whisper_outputs[i]
            if segment_text is None or not segment_text.get("text"):
                # Empty segment
                continue

            global_start = segment['start']
            global_end = segment['end']
            current_speaker = segment['speaker']
            text_segments = segment_text['offsets']

            for text_segment in text_segments:
                relative_start = text_segment['timestamp'][0]
                relative_end = text_segment['timestamp'][1]

                timestamp_start = global_start + relative_start
                timestamp_end = global_start + relative_end

                result.append({
                    "speaker": current_speaker,
                    "start": timestamp_start,
                    "end": timestamp_end,
                    "text": text_segment['text'].strip()
                })

        return result


    def __striped_legacy_call(self, inputs, diarization, group_by_speaker, kwargs_asr):
        ## This is the legacy method from upstream speechbox
        ## Mind the output format as they are not the same
        segments = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append({'segment': {'start': segment.start, 'end': segment.end},
                             'track': track,
                             'label': label})

        # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
        # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
        new_segments = []
        prev_segment = cur_segment = segments[0]

        for i in range(1, len(segments)):
            cur_segment = segments[i]

            # check if we have changed speaker ("label")
            if cur_segment["label"] != prev_segment["label"] and i < len(segments):
                # add the start/end times for the super-segment to the new list
                new_segments.append(
                    {
                        "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]},
                        "speaker": prev_segment["label"],
                    }
                )
                prev_segment = segments[i]

        # add the last segment(s) if there was no speaker change
        new_segments.append(
            {
                "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["end"]},
                "speaker": prev_segment["label"],
            }
        )

        asr_out = self.asr_pipeline(
            {"array": inputs, "sampling_rate": self.sampling_rate},
            return_timestamps=True,
            **kwargs_asr,
        )
        transcript = asr_out["chunks"]

        # get the end timestamps for each chunk from the ASR output
        total_duration = inputs.shape[0] / self.sampling_rate
        if transcript[-1]["timestamp"][-1] is None:
            transcript[-1]["timestamp"] = (transcript[-1]["timestamp"][0], total_duration)

        end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])

        segmented_preds = []

        # align the diarizer timestamps and the ASR timestamps
        for segment in new_segments:
            # get the diarizer end timestamp
            end_time = segment["segment"]["end"]

            # find the ASR end timestamp that is closest to the diarizer's end timestamp
            # only if the ending timestamp is not None
            if transcript and transcript[0]["timestamp"][-1] is not None:
                upto_idx = np.argmin(np.abs(end_timestamps - end_time))

                if group_by_speaker:
                    segmented_preds.append(
                        {
                            "speaker": segment["speaker"],
                            "text": "".join([chunk["text"] for chunk in transcript[:upto_idx + 1]]),
                            "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1]),
                        }
                    )
                else:
                    for i in range(upto_idx + 1):
                        segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

                # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
                transcript = transcript[upto_idx + 1:]
                end_timestamps = end_timestamps[upto_idx + 1:]

        return segmented_preds

    # Adapted from transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline.preprocess
    # (see https://github.com/huggingface/transformers/blob/238449414f88d94ded35e80459bb6412d8ab42cf/src/transformers/pipelines/automatic_speech_recognition.py#L417)
    def preprocess(self, inputs):
        if isinstance(inputs, str):
            if inputs.startswith("http://") or inputs.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                inputs = requests.get(inputs).content
            else:
                with open(inputs, "rb") as f:
                    inputs = f.read()

        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.sampling_rate).copy()

        if isinstance(inputs, dict):
            # Accepting `"array"` which is the key defined in `datasets` for better integration
            if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
                raise ValueError(
                    "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                    '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                    "containing the sampling_rate associated with that array"
                )

            _inputs = inputs.pop("raw", None)
            if _inputs is None:
                # Remove path which will not be used from `datasets`.
                inputs.pop("path", None)
                _inputs = inputs.pop("array", None)
            in_sampling_rate = inputs.pop("sampling_rate")
            inputs = _inputs
            if in_sampling_rate != self.sampling_rate:
                # @todo: Fix this stupid idea
                inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, self.sampling_rate).numpy()

        if not isinstance(inputs, np.ndarray):
            raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")

        if len(inputs.shape) != 1:
            print("We expect a single channel audio input for ASRDiarizePipeline so we downmix")
            inputs = np.mean(inputs, axis=0, keepdims=True)
        # Advanced preprocessing such as noise cancellation and other
        # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
        diarizer_inputs = F.resample(torch.from_numpy(inputs), self.sampling_rate,
                                     self.diarization_sampling_rate).float()
        diarizer_inputs = diarizer_inputs.unsqueeze(0)

        return inputs, diarizer_inputs
