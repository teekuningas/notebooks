import re
import os
import glob


def read_files(folder, prefix):
    contents = []
    for fname in sorted(os.listdir(folder)):
        if not fname.startswith(prefix):
            continue
        path = os.path.join(folder, fname)
        with open(path, 'r') as f:
            contents.append((fname, "".join(f.readlines())))
    sorted_contents = sorted(contents, key=lambda f: re.search(r'\d+', f[0]).group().zfill(3))
    return sorted_contents


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

def read_interview_data(base_path, interview_type):
    """
    Reads interview data from a nested directory structure.

    Args:
        base_path (str): The path to the directory containing the interview data,
                         structured as <uuid>/<type>/...
        interview_type (str): "observation" or "feedback".

    Returns:
        list: A list of tuples, where each tuple contains a metadata dictionary and the content as a string.
    """
    if interview_type == "observation":
        type_code = "0"
    elif interview_type == "feedback":
        type_code = "1"
    else:
        raise ValueError("Invalid interview_type. Must be 'observation' or 'feedback'.")

    search_pattern = os.path.join(base_path, "**", "*.txt")
    all_files = glob.glob(search_pattern, recursive=True)

    filtered_files = []
    for file_path in all_files:
        relative_path = os.path.relpath(file_path, base_path)
        path_parts = relative_path.split(os.sep)

        if len(path_parts) > 1 and path_parts[1] == type_code:
            filtered_files.append(file_path)

    file_contents = []
    for file_path in filtered_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content = strip_webvtt_to_plain_text(content)

                relative_path = os.path.relpath(file_path, base_path)
                path_parts = relative_path.split(os.sep)

                filename = os.path.basename(file_path)

                metadata = {
                    "user_id": path_parts[0],
                    "type": interview_type,
                    "rec_id": path_parts[2],
                    "timestamp": filename.replace(".txt", ""),
                    "filename": filename
                }

                file_contents.append((metadata, content))
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return file_contents

def filter_interview_simple(data):
    grouped_by_user = {}
    for metadata, content in data:
        user_id = metadata["user_id"]
        if user_id not in grouped_by_user:
            grouped_by_user[user_id] = []
        grouped_by_user[user_id].append((metadata, content))

    filtered_data = []
    for user_id, user_data in grouped_by_user.items():
        sorted_user_data = sorted(user_data, key=lambda x: x[0]["timestamp"])
        for metadata, content in sorted_user_data:
            stripped_text = content.strip()
            words = stripped_text.split()
            if len(words) >= 10 and len(set(words)) >= 10:
                filtered_data.append((metadata, content))
                break
    return filtered_data
