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