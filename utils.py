import re
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import ipywidgets as widgets
from IPython.display import display
from itertools import groupby


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

def display_interactive_dendrogram(vectors, items, item_key):
    # Helper function to calculate within-cluster sum of squares (WCSS) for a range of thresholds
    def calculate_wcss(vectors, Z, thresholds):
        vecs = np.array(vectors)
        wcss_values = []
        for t in thresholds:
            clusters = fcluster(Z, t, criterion='distance')
            wcss = 0
            for cluster_id in np.unique(clusters):
                cluster_points = vecs[clusters == cluster_id]
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)
            wcss_values.append(wcss)
        return wcss_values

    # Helper function to get the rendered text based on clusters and items
    def format_cluster_contents(clusters, items):
        cluster_dict = {}
        for cluster_id, item in zip(clusters, items):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(item)
            
        html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for cluster_id in sorted(cluster_dict.keys()):
            cluster_items = cluster_dict[cluster_id]
            html += f"<p><strong>Cluster {cluster_id}</strong> ({len(cluster_items)} items):<br>"
            html += ", ".join([item[item_key] for item in cluster_items])
            html += "</p>"
        html += "</div>"
        return html

    # Generate a static linkage-structure for the embedded items, used in clustering
    distance_matrix = pdist(vectors, metric='cosine')
    Z = linkage(distance_matrix, method='average')

    # Select sensible plot params
    min_threshold, max_threshold = 0.05, 0.8
    default_threshold = 0.25

    # Generate the knee-plot values
    thresholds = np.linspace(min_threshold, max_threshold, num=100)
    wcss_values = calculate_wcss(vectors, Z, thresholds)

    # Create a user-controllable slider
    threshold_slider = widgets.FloatSlider(
        value=default_threshold,
        min=min_threshold,
        max=max_threshold,
        step=(max_threshold - min_threshold) / 100,
        description='Threshold:',
        continuous_update=True
    )

    # Set defaults for the rendered text    
    cluster_info = widgets.Label(value="Number of clusters: 0")
    cluster_contents = widgets.HTML(value="")

    # Create the interactive plot
    # This is called everytime we change the slider
    def update_plot(threshold):

        # Create figure for the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # First update the tree-plot
        dendrogram(Z, ax=ax1, no_labels=True)
        ax1.axhline(y=threshold, color='r', linestyle='--')

        # Then update the WCSS plot
        ax2.plot(thresholds, wcss_values, label='WCSS')
        ax2.axvline(x=threshold, color='r', linestyle='--')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('WCSS')

        # Finally, show the clusters for this specific threshold in text below the plots.
        clusters = fcluster(Z, threshold, criterion='distance')
        n_clusters = len(np.unique(clusters))
        cluster_info.value = f"Number of clusters: {n_clusters}"
        cluster_contents.value = format_cluster_contents(clusters, items)
        
        # Ensure proper layout
        plt.tight_layout()
        plt.show()

    # Finally, create and show interactive clustering widget.
    interactive_plot = widgets.interactive(update_plot, threshold=threshold_slider)
    display(widgets.VBox([
        threshold_slider,
        cluster_info,
        interactive_plot.children[-1],
        cluster_contents
    ]))

def format_raw_output(items, item_key, run_id, prefix, id_key='rec_id'):
    raw_output_str = ""
    preview_output_str = ""
    
    id_key_name = "Rec ID Anonymized" if id_key == 'rec_id_anonymized' else "Rec ID"

    unique_rec_ids = sorted(list(set(i[id_key] for i in items if id_key in i)))

    for i, rec_id in enumerate(unique_rec_ids):
        
        items_for_rec = [it for it in items if id_key in it and it[id_key] == rec_id]
        items_by_iter = {k: list(v) for k, v in groupby(sorted(items_for_rec, key=lambda x: x['iter_idx']), key=lambda x: x['iter_idx'])}
        
        rec_str = f"{id_key_name}: {rec_id}\n"
        rec_str += "-----------------\n"
        
        for iter_idx, iter_items in sorted(items_by_iter.items()):
            rec_str += f"Iter {iter_idx}:\n"
            for item in iter_items:
                rec_str += f"  - {item_key.capitalize()}: {item[item_key]}\n"
                rec_str += f"    Explanation: {item['explanation']}\n"
        rec_str += "\n"
        
        raw_output_str += rec_str
        if i < 10:
            preview_output_str += rec_str

    output_dir = f"output/{prefix}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{prefix}_raw_{id_key}.txt"
    
    if len(unique_rec_ids) > 10:
        preview_output_str += f"\n... [Showing 10 of {len(unique_rec_ids)} records. Full content in '{filename}']"

    with open(filename, "w") as f:
        f.write(raw_output_str)
        
    return preview_output_str

def format_summary_output(final_items, clusters, item_key, run_id, prefix, id_key='rec_id'):
    summary_str = ""
    preview_summary_str = ""
    id_key_name = "Rec ID Anonymized" if id_key == 'rec_id_anonymized' else "Rec ID"
    
    for idx, (final_item, cluster) in enumerate(zip(final_items, clusters)):
        cluster_summary = f"{item_key.capitalize()}: {final_item[item_key]}\n"
        cluster_summary += "-------------------------\n"
        for item in cluster:
            if id_key in item:
                cluster_summary += f"  - Explanation: {item['explanation']}\n"
                cluster_summary += f"    {id_key_name}: {item[id_key]}, Iteration: {item['iter_idx']}\n"
        cluster_summary += "\n"
        
        summary_str += cluster_summary
        if idx < 10:
            preview_summary_str += cluster_summary

    output_dir = f"output/{prefix}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/{prefix}_summary_{id_key}.txt"

    if len(final_items) > 10:
        preview_summary_str += f"\n... [Showing 10 of {len(final_items)} summaries. Full content in '{filename}']"

    with open(filename, "w") as f:
        f.write(summary_str)
        
    return preview_summary_str