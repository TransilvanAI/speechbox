from .utils import seconds_to_srt_timestamp
from datetime import timedelta



def dialog_to_srt(dialog):
    srt_output = []
    for i, entry in enumerate(dialog):
        start = seconds_to_srt_timestamp(entry['start'])
        end = seconds_to_srt_timestamp(entry['end'])
        text = entry['text']
        srt_block = f"{i + 1}\n{start} --> {end}\n{entry['speaker']}: {text}\n\n"
        srt_output.append(srt_block)
    return ''.join(srt_output)




def seconds_to_vtt_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"


def dialog_json_to_vtt(dialog_json):
    vtt_lines = ["WEBVTT\n"]

    for entry in dialog_json:
        speaker = entry.get("speaker", "Unknown Speaker")
        start = seconds_to_vtt_timestamp(entry["start"])
        end = seconds_to_vtt_timestamp(entry["end"])
        text = entry["text"]

        vtt_lines.append(f"{start} --> {end}")
        vtt_lines.append(f"{speaker}: {text}\n")

    return "\n".join(vtt_lines)