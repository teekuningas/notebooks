import re
import os


def read_files(folder, prefix):
    contents = []
    for fname in sorted(os.listdir(folder)):
        if not fname.startswith(prefix):
            continue
        path = os.path.join(folder, fname)
        with open(path, 'r') as f:
            contents.append((fname, "".join(f.readlines())))
    return contents


def strip_webvtt_to_plain_text(webvtt_string):
    # Remove WEBVTT header
    webvtt_string = re.sub(r'^WEBVTT\s*\n', '', webvtt_string, flags=re.MULTILINE)
    
    # Remove timestamps
    webvtt_string = re.sub(r'\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}.\d{3}\s*\n', '', webvtt_string)
    
    # Remove speaker tags
    webvtt_string = re.sub(r'\[SPEAKER_\d{2}\]:\s', '', webvtt_string)
    
    # Replace multiple newlines with double newline (for better readability)
    webvtt_string = re.sub(r'\n\n+', '\n\n', webvtt_string)
    
    # Strip leading/trailing whitespace from start and end
    webvtt_string = webvtt_string.strip()
    
    return webvtt_string