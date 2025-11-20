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
from utils import display_interactive_dendrogram, format_raw_output, format_summary_output
from uuid import uuid4

# Start by reading the texts of interest from the file system
# contents = read_files(folder="data/luontokokemus_simulaatio", prefix="interview")
contents = read_interview_data("data/birdinterview", "observation")

# And filter to a sensible subset
contents = filter_interview_simple(contents)

#contents = contents[:10]

rec_id_map = {meta['rec_id']: str(uuid4()) for meta, _ in contents}
for meta, text in contents:
    meta['rec_id_anonymized'] = rec_id_map[meta['rec_id']]

print(f"len(contents): {len(contents)}")

## Print to check that texts are correctly read
#for meta, text in contents:
#    print(f"{meta['rec_id']}:\n\n")
#    print(f"{text}\n\n")

# %%
import json
import requests
from pprint import pprint
from uuid import uuid4

from llm import generate_simple

run_id = str(uuid4())[:8]

# Now, generate a locationbook for each text. To mitigate randomness, we actually generate multiple locationbooks for each text, and afterwards combine all locationbooks of all texts into one locationbook with clustering.

# Specify a machine-readable output format for a single locationbook (a list of location-explanation pairs)
output_format = {
    "type": "object",
    "properties": {
        "locations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string"
                    },
                    "explanation": {
                        "type": "string"
                    }
                },
                "required": [
                    "location",
                    "explanation"
                ]
            }
        }
    },
    "required": [
      "locations"
    ]
}

# For every text, generate {n_iter} locationbooks. 
n_iter = 1

locations = []
for rec_idx, (meta, text) in enumerate(contents):
    print(f"Generating locationbook {rec_idx+1} for {meta['rec_id']}..")
    for iter_idx in range(n_iter):
        seed = iter_idx + 1

        while True: # Retry indefinitely until success
            try:
                # It is easier for llms to do the locationbook in two steps: first request a free-formatted locationbook and then request it in the correct format. 
                # This also allows different roles: we could use a reasoning model (which is bad at formatting) to do the first step and a formatting model (which is bad at reasoning) to do the second step.

                # Define the instruction for the first step:
                instruction = """
                Olet laadullisen tutkimuksen avustaja. Tehtäväsi on tunnistaa tekstistä viittaukset ympäristöihin ja paikkoihin.

                Perustele selkeästi jokaiselle paikalle erikseen millä tavoin se ilmenee tekstissä.

                Vältä lyhenteitä.

                Vastaa suomeksi.
                """ 
                
                # Generate the intermediate result using the text, the instruction and a unique seed (to get a different result each time):
                result = generate_simple(instruction, text, seed=seed, provider="llamacpp", timeout=300)

                # Extract the result
                content = result

                # Define the instruction for the formatting task: 
                instruction = """
                Muotoile annettu vapaamuotoinen lista pyydettyyn JSON-muotoon.
                """

                # Generate the machine-readable result:
                result = generate_simple(instruction, content, seed=10, output_format=output_format, provider="llamacpp", timeout=300)

                # Extract the machine-readable result.
                generated_locations = json.loads(result)['locations']

                # Store the single result.
                for loc in generated_locations:
                    locations.append({
                        "location": loc["location"],
                        "explanation": loc["explanation"],
                        "rec_id": meta['rec_id'],
                        "rec_id_anonymized": meta['rec_id_anonymized'],
                        "iter_idx": iter_idx
                    })
                break  # Success, exit while loop
            except Exception as e:
                seed += 1
                print(f"An error occurred: {e}. Retrying with new seed {seed}...")

# Now locations variable includes all location-explanation pairs from all texts and all iterations.
# Create a human-readable and machine-parsable raw output.
preview = format_raw_output(locations, 'location', run_id, 'paikat', id_key='rec_id')
print(preview)
preview = format_raw_output(locations, 'location', run_id, 'paikat', id_key='rec_id_anonymized')
print(preview)


# %%
import numpy as np

from llm import embed

# In the following steps, we 
# 1) Flatten all the locationbooks into a single list of locations
# 2) The locations are embedded into a "semantic space" 
# 3) The embedded locations are clustered (put into groups) and the smallest clusters are discarded. 
# 4) The locations in a cluster are combined so that each cluster . 
# This process allows us to find the most reliable locations of all texts. Of course, this has a trade-off and manual inspection is recommended.

# Embed each location into the "semantic space".
vectors = []
for location in locations:
    result = embed(location['location'].replace("*", ""), provider="llamacpp")
    vectors.append(result)

print(f"{len(vectors)} locations embedded.")

# %%
display_interactive_dendrogram(vectors, locations, 'location')

# %%
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import os
# Here we actually apply the threshold, create the clusters, and generate a representative location for each.

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
for cluster_id, location in zip(clusters, locations):
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []
    cluster_dict[cluster_id].append(location)
location_clusters = sorted([cluster[1] for cluster in cluster_dict.items()], key=lambda x: len(x))

# Discard smallest clusters (size < {min_size})
location_clusters = [cluster for cluster in location_clusters if len(cluster) >= min_size]

# Now we have clusters, but we actually wanted a locationbook. We will ask llm to pick or create a represenative location for each of the clusters.

# Specify a machine-readable output format
output_format = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string"
        }
    },
    "required": [
      "location"
    ]
}

# Go through each of the clusters, and generate a representative
final_locations = []
for idx, cluster in enumerate(location_clusters):
    
    # Define a instruction:
    
    instruction = """
    Olet laadullisen tutkimuksen avustaja.
    Tehtäväsi on tiivistää lista samankaltaisia paikkoja yhdeksi edustavaksi paikaksi.
    Valitse olemassaolevista paikoista paras.
    """

    # For data, we set a comma-separated list of cluster elements.
    data = ", ".join([item['location'] for item in cluster])

    # Send the request to llm.
    result = generate_simple(instruction, data, seed=10, output_format=output_format, provider="llamacpp")

    # Extract the location
    location = json.loads(result)

    # And store it
    final_locations.append(location)

# Print the final locationbook and save to file.
output_str = "Final locations:\n"
preview_str = "Final locations:\n"
for idx, location in enumerate(final_locations):
    line = f"{idx+1}: {location} ({', '.join([item['location'] for item in location_clusters[idx]])})\n"
    output_str += line
    if idx < 10:
        preview_str += line

output_dir = f"output/paikat/{run_id}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/paikat_short.txt"

if len(final_locations) > 10:
    preview_str += f"\n... [Showing 10 of {len(final_locations)} locations. Full content in '{filename}']"

output_str += "\nIn a format for easy copying:\n"
output_str += str(list(dict.fromkeys(reversed([locationdict['location'] for locationdict in final_locations]))))

print(preview_str)

with open(filename, "w") as f:
    f.write(output_str)

# %%
# Create a summary document
preview = format_summary_output(final_locations, location_clusters, 'location', run_id, 'paikat', id_key='rec_id')
print(preview)
preview = format_summary_output(final_locations, location_clusters, 'location', run_id, 'paikat', id_key='rec_id_anonymized')
print(preview)

# %%
