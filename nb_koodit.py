# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from utils import read_files
from utils import read_interview_data
from utils import strip_webvtt_to_plain_text

from utils import filter_interview_simple

# Start by reading the texts of interest from the file system
# contents = read_files(folder="data/luontokokemus_simulaatio", prefix="interview")
contents = read_interview_data("data/birdinterview", "observation")

# And filter to a sensible subset
contents = filter_interview_simple(contents)

contents = contents[:5]

# Convert to (rec_id, content) tuples for now
contents = [(meta["rec_id"], text) for meta, text in contents]

print(f"len(contents): {len(contents)}")

## Print to check that texts are correctly read
#for rec_id, text in contents:
#    print(f"{rec_id}:\n\n")
#    print(f"{text}\n\n")

# %%
import json
import requests
from pprint import pprint

from llm import generate_simple

# Now, generate a codebook for each text. To mitigate randomness, we actually generate multiple codebooks for each text, and afterwards combine all codebooks of all texts into one codebook with clustering.

# Specify a machine-readable output format for a single codebook (a list of code-explanation pairs)
output_format = {
    "type": "object",
    "properties": {
        "codes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string"
                    },
                    "explanation": {
                        "type": "string"
                    }
                },
                "required": [
                    "code",
                    "explanation"
                ]
            }
        }
    },
    "required": [
      "codes"
    ]
}

# For every text, generate {n_iter} codebooks. 
n_iter = 1

codes = []
for rec_idx, (rec_id, text) in enumerate(contents):
    print(f"Generating codebook {rec_idx+1} for {rec_id}..")
    for iter_idx in range(n_iter):
        seed = iter_idx + 1

        while True: # Retry indefinitely until success
            try:
                # It is easier for llms to do the codebook in two steps: first request a free-formatted codebook and then request it in the correct format. 
                # This also allows different roles: we could use a reasoning model (which is bad at formatting) to do the first step and a formatting model (which is bad at reasoning) to do the second step.

                # Define the instruction for the first step:
                instruction = """
                Olet laadullisen tutkimuksen avustaja. Tehtäväsi on tunnistaa tekstistä teemat ja käsitteet.

                Muodosta näistä teemoista koodikirja. Jokaisella koodilla tulee olla perustelu.

                Vältä lyhenteitä.

                Vastaa suomeksi.
                """ 
                
                # Generate the intermediate result using the text, the instruction and a unique seed (to get a different result each time):
                result = generate_simple(instruction, text, seed=seed, provider="llamacpp", timeout=300)

                # Extract the result
                content = result

                # Define the instruction for the formatting task: 
                instruction = """
                Muotoile annettu vapaamuotoinen koodikirja pyydettyyn JSON-muotoon.
                """

                # Generate the machine-readable result:
                result = generate_simple(instruction, content, seed=10, output_format=output_format, provider="llamacpp", timeout=300)

                # Extract the machine-readable result.
                generated_codes = json.loads(result)['codes']

                # Store the single result.
                for c in generated_codes:
                    codes.append({
                        "code": c["code"],
                        "explanation": c["explanation"],
                        "rec_id": rec_id,
                        "iter_idx": iter_idx
                    })
                break  # Success, exit while loop
            except Exception as e:
                seed += 1
                print(f"An error occurred: {e}. Retrying with new seed {seed}...")

# Now codes variable includes all code-explanation pairs from all texts and all iterations.
pprint(codes)


# %%
import numpy as np

from llm import embed

# In the following steps, we 
# 1) Flatten all the codebooks into a single list of codes
# 2) The codes are embedded into a "semantic space" 
# 3) The embedded codes are clustered (put into groups) and the smallest clusters are discarded. 
# 4) The codes in a cluster are combined so that each cluster . 
# This process allows us to find the most reliable codes of all texts. Of course, this has a trade-off and manual inspection is recommended.

# Embed each code into the "semantic space".
vectors = []
for code in codes:
    result = embed(code['code'].replace("*", ""), provider="llamacpp")
    vectors.append(result)

print(f"{len(vectors)} codes embedded.")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import ipywidgets as widgets
from IPython.display import display

# Then the clustering. The folllab/tree/notebooks/nb_analyysi_koodit.ipynbowing code creates a visualization of the clustering as a tree and creates a user-controlled slider to change the cut-off threshold.
# The resulting clusters are written below the plots. Here, one can experiment with different thresholds. The threshold is then set in the next step.

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

# Helper function to get the rendered text based on clusters and codes
def format_cluster_contents(clusters, codes):
    cluster_dict = {}
    for cluster_id, code in zip(clusters, codes):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(code)
        
    html = "<div style='max-height: 400px; overflow-y: auto;'>"
    for cluster_id in sorted(cluster_dict.keys()):
        items = cluster_dict[cluster_id]
        html += f"<p><strong>Cluster {cluster_id}</strong> ({len(items)} items):<br>"
        html += ", ".join([item['code'] for item in items])
        html += "</p>"
    html += "</div>"
    return html

# Generate a static linkage-structure for the embedded codes, used in clustering
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
    cluster_contents.value = format_cluster_contents(clusters, codes)
    
    # Ensure proper layout
    plt.tight_layout()
    plt.show()

# Finall, create and show interactive clustering widget.
interactive_plot = widgets.interactive(update_plot, threshold=threshold_slider)
display(widgets.VBox([
    threshold_slider,
    cluster_info,
    interactive_plot.children[-1],
    cluster_contents
]))

# %%
from uuid import uuid4
# Here we actually apply the threshold and create the clusters.

# The threshold-parameter for the clustering (bigger threshold means bigger clusters. Here, smaller values are probably better.)
threshold = 0.25

# Discard clusters with less than {min_size} elements.
min_size = 2

# Compute the clusters with the selected threshold
distance_matrix = pdist(vectors, metric='cosine')
Z = linkage(distance_matrix, method='average')
clusters = fcluster(Z, threshold, criterion='distance')

# Reorganize to a simple list of clusters
cluster_dict = {}
for cluster_id, code in zip(clusters, codes):
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []
    cluster_dict[cluster_id].append(code)
code_clusters = sorted([cluster[1] for cluster in cluster_dict.items()], key=lambda x: len(x))

# Write all clusters to a file
output_str = ""
for idx, cluster in enumerate(code_clusters):
    output_str += f"{idx+1}: {[item['code'] for item in cluster]}\n"
    
with open(f"output/koodit_clusters_{str(uuid4())[:8]}.txt", "w") as f:
    f.write(output_str)

# Discard smallest clusters (size < {min_size})
code_clusters = [cluster for cluster in code_clusters if len(cluster) >= min_size]

# Finally print the resulting clusters to check
for idx, cluster in enumerate(code_clusters):
    print(f"{idx+1}: {[item['code'] for item in cluster]}")

# %%
from llm import generate_simple
from uuid import uuid4

# Now we have clusters, but we actually wanted a codebook. We will ask llm to pick or create a represenative code for each of the clusters.

# Specify a machine-readable output format
output_format = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string"
        }
    },
    "required": [
      "code"
    ]
}

# Go through each of the clusters, and generate a representative
final_codes = []
for idx, cluster in enumerate(code_clusters):
    
    # Define a instruction:
    
    instruction = """
    Olet laadullisen tutkimuksen avustaja.
    Tehtäväsi on tiivistää lista samankaltaisia koodeja yhdeksi edustavaksi koodiksi.
    Valitse olemassaolevista koodeista paras.
    """

    # For data, we set a comma-separated list of cluster elements.
    data = ", ".join([item['code'] for item in cluster])

    # Send the request to llm.
    result = generate_simple(instruction, data, seed=10, output_format=output_format, provider="llamacpp")

    # Extract the code
    code = json.loads(result)

    # And store it
    final_codes.append(code)

# Print the final codebook and save to file.
output_str = "Final codes:\n"
for idx, code in enumerate(final_codes):
    output_str += f"{idx+1}: {code} ({', '.join([item['code'] for item in code_clusters[idx]])})\n"
output_str += "\nIn a format for easy copying:\n"
output_str += str(list(dict.fromkeys(reversed([codedict['code'] for codedict in final_codes]))))

print(output_str)

with open(f"output/koodit_{str(uuid4())[:8]}.txt", "w") as f:
    f.write(output_str)

# %%
from uuid import uuid4

# Create a summary document
summary_str = ""
for final_code, cluster in zip(final_codes, code_clusters):
    summary_str += f"Code: {final_code['code']}\n"
    summary_str += "-------------------------\n"
    for item in cluster:
        summary_str += f"  - Explanation: {item['explanation']}\n"
        summary_str += f"    Rec ID: {item['rec_id']}, Iteration: {item['iter_idx']}\n"
    summary_str += "\n"

print(summary_str)

with open(f"output/koodit_summary_{str(uuid4())[:8]}.txt", "w") as f:
    f.write(summary_str)

# %%
