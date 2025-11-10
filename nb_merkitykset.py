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

contents = contents[0:3]

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
n_iter = 2

meanings = []
for rec_idx, (meta, text) in enumerate(contents):
    print(f"Generating meaningbook {rec_idx+1} for {meta['rec_id']}..")
    for iter_idx in range(n_iter):
        seed = iter_idx + 1

        while True: # Retry indefinitely until success
            try:
                # It is easier for llms to do the meaningbook in two steps: first request a free-formatted meaningbook and then request it in the correct format.
                # This also allows different roles: we could use a reasoning model (which is bad at formatting) to do the first step and a formatting model (which is bad at reasoning) to do the second step.

                # Define the instruction for the first step:
                instruction = """
                Olet laadullisen tutkimuksen avustaja. Tehtäväsi on tunnistaa ja listata tekstistä kaikki emotionaaliset merkitykset.

                Perustele selkeästi jokaiselle merkitykselle erikseen millä tavoin se ilmenee tekstissä.

                Vastaa suomeksi.
                """

                # Generate the intermediate result using the text, the instruction and a unique seed (to get a different result each time):
                result = generate_simple(instruction, text, seed=seed, provider="llamacpp", timeout=300)

                # Extract the result
                content = result

                # Define the instruction for the formatting task:
                instruction = """
                Muotoile annettu vapaamuotoinen emotionaalisista merkityksistä koostuva lista pyydettyyn JSON-muotoon.
                """

                # Generate the machine-readable result:
                result = generate_simple(instruction, content, seed=10, output_format=output_format, provider="llamacpp", timeout=300)

                # Extract the machine-readable result.
                generated_meanings = json.loads(result)['meanings']

                # Store the single result.
                for m in generated_meanings:
                    meanings.append({
                        "meaning": m["meaning"],
                        "explanation": m["explanation"],
                        "rec_id": meta['rec_id'],
                        "rec_id_anonymized": meta['rec_id_anonymized'],
                        "iter_idx": iter_idx
                    })
                break  # Success, exit while loop
            except Exception as e:
                seed += 1
                print(f"An error occurred: {e}. Retrying with new seed {seed}...")

# Now meanings variable includes all meaning-explanation pairs from all texts and all iterations.
# Create a human-readable and machine-parsable raw output.
preview = format_raw_output(meanings, 'meaning', run_id, 'merkitykset', id_key='rec_id')
print(preview)
preview = format_raw_output(meanings, 'meaning', run_id, 'merkitykset', id_key='rec_id_anonymized')
print(preview)


# %%
import numpy as np

from llm import embed

# In the following steps, we 
# 1) Flatten all the meaningbooks into a single list of meanings
# 2) The meanings are embedded into a "semantic space" 
# 3) The embedded meanings are clustered (put into groups) and the smallest clusters are discarded. 
# 4) The meanings in a cluster are combined so that each cluster has a representative meaning.
# This process allows us to find the most reliable meanings of all texts. Of course, this has a trade-off and manual inspection is recommended.

# Embed each meaning into the "semantic space".
vectors = []
for meaning in meanings:
    result = embed(meaning['meaning'].replace("*", ""), provider="llamacpp")
    vectors.append(result)

print(f"{len(vectors)} meanings embedded.")

# %%
display_interactive_dendrogram(vectors, meanings, 'meaning')

# %%
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import os
# Here we actually apply the threshold, create the clusters, and generate a representative meaning for each.

# The threshold-parameter for the clustering (bigger threshold means bigger clusters. Here, smaller values are probably better.)
threshold = 0.20

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

# Discard smallest clusters (size < {min_size})
meaning_clusters = [cluster for cluster in meaning_clusters if len(cluster) >= min_size]

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
    Valitse merkitykseksi joku listassa esiintyvistä merkityksistä.
    """

    # For data, we set a comma-separated list of cluster elements.
    data = ", ".join([item['meaning'] for item in cluster])

    # Send the request to llm.
    result = generate_simple(instruction, data, seed=10, output_format=output_format, provider="llamacpp")

    # Extract the meaning
    meaning = json.loads(result)

    # And store it
    final_meanings.append(meaning)

# Print the final meaningbook and save to file.
output_str = "Final meanings:\n"
preview_str = "Final meanings:\n"
for idx, meaning in enumerate(final_meanings):
    line = f"{idx+1}: {meaning} ({', '.join([item['meaning'] for item in meaning_clusters[idx]])})\n"
    output_str += line
    if idx < 10:
        preview_str += line

output_dir = f"output/merkitykset/{run_id}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/merkitykset_short.txt"

if len(final_meanings) > 10:
    preview_str += f"\n... [Showing 10 of {len(final_meanings)} meanings. Full content in '{filename}']"

output_str += "\nIn a format for easy copying:\n"
output_str += str(list(dict.fromkeys(reversed([meaningdict['meaning'] for meaningdict in final_meanings]))))

print(preview_str)

with open(filename, "w") as f:
    f.write(output_str)

# %%
# Create a summary document
preview = format_summary_output(final_meanings, meaning_clusters, 'meaning', run_id, 'merkitykset', id_key='rec_id')
print(preview)
preview = format_summary_output(final_meanings, meaning_clusters, 'meaning', run_id, 'merkitykset', id_key='rec_id_anonymized')
print(preview)

# %%
