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

# Convert to (rec_id, content) tuples for now
contents = [(meta["rec_id"], text) for meta, text in contents]

contents = contents[:5]

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

# Now, generate a meaningbook for each text. To mitigate randomness, we actually generate multiple meaningbooks for each text, and afterwards combine all meaningbooks of all texts into one meaningbook with clustering.

# Specify a machine-readable output format for a single meaningbook (a list of meaning-explanation pairs)
output_format = {
    "type": "object",
    "properties": {
        "meanings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "meaning": {
                        "type": "string"
                    },
                    "explanation": {
                        "type": "string"
                    }
                },
                "required": [
                    "meaning",
                    "explanation"
                ]
            }
        }
    },
    "required": [
      "meanings"
    ]
}

# For every text, generate {n_iter} meaningbooks. 
n_iter = 1

meaninglists = []
for idx, (rec_id, text) in enumerate(contents):
    print(f"Generating meaningbook {idx+1} for {rec_id}..")
    subresults = []
    for idx in range(n_iter):

        # It is easier for llms to do the meaningbook in two steps: first request a free-formatted meaningbook and then request it in the correct format. 
        # This also allows different roles: we could use a reasoning model (which is bad at formatting) to do the first step and a formatting model (which is bad at reasoning) to do the second step.

        # Define the instruction for the first step:
        instruction = """
        Olet laadullisen tutkimuksen avustaja. Tehtäväsi on tunnistaa tekstistä kaikki emotionaaliset merkitykset.

        Listaa emotionaaliset merkitykset. Jokaisella emotionaalisella merkityksella tulee olla perustelu.

        Vastaa suomeksi.
        """ 
        
        # Generate the intermediate result using the text, the instruction and a unique seed (to get a different result each time):
        result = generate_simple(instruction, text, seed=idx, provider="llamacpp")

        # Extract the result
        content = result

        # Define the instruction for the formatting task: 
        instruction = """
        Muotoile annettu vapaamuotoinen emotionaalisista merkityksista koostuva lista pyydettyyn JSON-muotoon.
        """

        # Generate the machine-readable result:
        result = generate_simple(instruction, content, seed=10, output_format=output_format, provider="llamacpp")

        # Extract the machine-readable result.
        meanings = json.loads(result)['meanings']

        # Store the single result.
        subresults.append(meanings)

    # Store the results of all iterations of a single text.
    meaninglists.append((rec_id, subresults))

# Now meaninglists variable includes {n_iter} meaningbooks for each of the texts. Print them to see.
for rec_id, subresults in meaninglists:
    print(f"{rec_id}:\n")
    for idx, meaninglist in enumerate(subresults):
        print(f"Iteraatio {idx+1}\n")
        pprint(meaninglist)
        print("")


# %%
import numpy as np

from llm import embed

# In the following steps, we 
# 1) Flatten all the meaningbooks into a single list of meanings
# 2) The meanings are embedded into a "semantic space" 
# 3) The embedded meanings are clustered (put into groups) and the smallest clusters are discarded. 
# 4) The meanings in a cluster are combined so that each cluster has a representative meaning.
# This process allows us to find the most reliable meanings of all texts. Of course, this has a trade-off and manual inspection is recommended.

# First, flatten the meaningbooks into a single list of meanings.
meanings = []
for name, textvalues in meaninglists:
    for iteration in textvalues:
        for meaningdict in iteration:
            meanings.append(meaningdict['meaning'].replace("*", ""))

# Embed each meaning into the "semantic space".
vectors = []
for meaning in meanings:
    result = embed(meaning, provider="llamacpp")
    vectors.append(result)

print(f"{len(vectors)} meanings embedded.")

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
import ipywidgets as widgets
from IPython.display import display

# Then the clustering. The following code creates a visualization of the clustering as a tree and creates a user-controlled slider to change the cut-off threshold.
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

# Helper function to get the rendered text based on clusters and meanings
def format_cluster_contents(clusters, meanings):
    cluster_dict = {}
    for cluster_id, meaning in zip(clusters, meanings):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(meaning)
        
    html = "<div style='max-height: 400px; overflow-y: auto;'>"
    for cluster_id in sorted(cluster_dict.keys()):
        items = cluster_dict[cluster_id]
        html += f"<p><strong>Cluster {cluster_id}</strong> ({len(items)} items):<br>"
        html += ", ".join(items)
        html += "</p>"
    html += "</div>"
    return html

# Generate a static linkage-structure for the embedded meanings, used in clustering
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
    cluster_contents.value = format_cluster_contents(clusters, meanings)
    
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
for cluster_id, meaning in zip(clusters, meanings):
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []
    cluster_dict[cluster_id].append(meaning)
meaning_clusters = sorted([cluster[1] for cluster in cluster_dict.items()], key=lambda x: len(x))

# Write all clusters to a file
output_str = ""
for idx, cluster in enumerate(meaning_clusters):
    output_str += f"{idx+1}: {cluster}\n"
    
with open(f"output/clusters_{str(uuid4())[:8]}.txt", "w") as f:
    f.write(output_str)

# Discard smallest clusters (size < {min_size})
meaning_clusters = [cluster for cluster in meaning_clusters if len(cluster) >= min_size]

# Finally print the resulting clusters to check
for idx, cluster in enumerate(meaning_clusters):
    print(f"{idx+1}: {cluster}")


# %%
from llm import generate_simple
from uuid import uuid4

# Now we have clusters, but we actually wanted a meaningbook. We will ask llm to pick or create a represenative meaning for each of the clusters.

# Specify a machine-readable output format
output_format = {
    "type": "object",
    "properties": {
        "meaning": {
            "type": "string"
        }
    },
    "required": [
      "meaning"
    ]
}

# Go through each of the clusters, and generate a representative
final_meanings = []
for idx, cluster in enumerate(meaning_clusters):
    
    # Define a instruction:
    
    instruction = """
    Olet laadullisen tutkimuksen avustaja.
    Tehtäväsi on tiivistää lista samankaltaisia merkityksiä yhdeksi edustavaksi merkitykseksi.
    Valitse olemassa olevista merkityksistä paras.
    """

    # For data, we set a comma-separated list of cluster elements.
    data = ", ".join(cluster)

    # Send the request to llm.
    result = generate_simple(instruction, data, seed=10, output_format=output_format, provider="llamacpp")

    # Extract the meaning
    meaning = json.loads(result)

    # And store it
    final_meanings.append(meaning)

# Print the final meaningbook and save to file.
output_str = "Final meanings:\n"
for idx, meaning in enumerate(final_meanings):
    output_str += f"{idx+1}: {meaning} ({', '.join(meaning_clusters[idx])})\n"
output_str += "\nIn a format for easy copying:\n"
output_str += str(list(dict.fromkeys(reversed([meaningdict['meaning'] for meaningdict in final_meanings]))))

print(output_str)

with open(f"output/merkitykset_{str(uuid4())[:8]}.txt", "w") as f:
    f.write(output_str)

# %%
