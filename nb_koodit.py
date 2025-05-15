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

# %% [markdown]
# Mahdollisia tarkentavia tutkimuskysymyksiä koodikirjaa ajatellen:
#
# PAIKKA: miten ympäristöä ja luontoa kuvataan, mitä elementtejä mainitaan?  
# TOIMIJAT: kenestä/mistä puhutaan (ihmiset, eläimet, ei-inhimilliset toimijat), miten vuorovaikutusta kuvataan (ollaanko yksin, seurassa)?  
# AKTIVITEETIT/TOIMINTA: mitä luonnossa tehdään, miten siellä ollaan/liikutaan?  
# TAPAHTUMAT: mistä kerrotaan, mitä havainnoinnin aikana on tapahtunut?  
# MOTIIVIT JA TAVOITTEET: miksi luonnossa ollaan, mitä siellä halutaan tehdä ja saavuttaa?  
# TUNTEET: miltä osallistujasta tuntuu ja miten hän puhuu omista tuntemuksistaan ja aistimuksistaan? 

# %%
from utils import read_files
from utils import strip_webvtt_to_plain_text

# Start by reading the texts of interest from the file system
contents = read_files(folder="data/luontokokemus_simulaatio", prefix="interview")
# contents = read_files(folder="data/linnut", prefix="nayte")

# Remove timestamps if present
contents = [(fname, strip_webvtt_to_plain_text(text)) for fname, text in contents]

# Print to check that texts are correctly read
for fname, text in contents:
    print(f"{fname}:\n\n")
    print(f"{text}\n\n")

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
n_iter = 2

codelists = []
for fname, text in contents:
    subresults = []
    for idx in range(n_iter):

        # It is easier for llms to do the codebook in two steps: first request a free-formatted codebook and then request it in the correct format. 
        # This also allows different roles: we could use a reasoning model (which is bad at formatting) to do the first step and a formatting model (which is bad at reasoning) to do the second step.

        # Define the instruction for the first step:
        instruction = """
        Olet laadullisen tutkimuksen avustaja. Tarkoituksesi on etsiä seuraavasta tekstistä olennaisimmat käsitteet eli koodit.
        
        Lue teksti huolella ja anna vastaukseksi koodit ja jokaiselle koodille jokin perustelu. Yritä valita vain tärkeimmät ja ytimekkäimmät koodit, suosi yksisanaisia koodeja. Vastaa suomeksi.
        """
        
        # # An alternative for a more specific question:
        # instruction = """
        # Olet laadullisen tutkimuksen avustaja. Tarkoituksesi on etsiä seuraavasta tekstistä olennaisimmat käsitteet eli koodit, kuitenkin niin että ne liittyvät erityisesti seuraavaan tutkimuskysymykseen:
        # 
        # TAPAHTUMAT: mistä kerrotaan, mitä havainnoinnin aikana on tapahtunut?
        # 
        # Lue teksti huolella ja anna vastaukseksi koodit ja jokaiselle koodille jokin perustelu. Keskity tutkimuskysymykseen liittyviin koodeihin. Yritä valita vain tärkeimmät ja ytimekkäimmät koodit, suosi yksisanaisia koodeja. Vastaa suomeksi.
        # """   
        
        # Generate the intermediate result using the text, the instruction and a unique seed (to get a different result each time):
        result = generate_simple(instruction, text, seed=idx)

        # Extract the result
        content = result['message']['content']

        # Define the instruction for the formatting task: 
        instruction = """
        Olet laadullisen tutkimuksen avustaja. Ota alla oleva vapaamuotoisesti esitetty koodikirja ja muotoile se uudestaan pyydettyyn muotoon.
        """

        # Generate the machine-readable result:
        result = generate_simple(instruction, content, seed=10, output_format=output_format)

        # Extract the machine-readable result.
        codes = json.loads(result['message']['content'])['codes']
        
        # Store the single result.
        subresults.append(codes)
        
    # Store the results of all iterations of a single text.
    codelists.append((fname, subresults))

# Now codelists variable includes {n_iter} codebooks for each of the texts. Print them to see.
for fname, subresults in codelists:
    print(f"{fname}:\n")
    for idx, codelist in enumerate(subresults):
        print(f"Iteraatio {idx+1}\n")
        pprint(codelist)
        print("")


# %%
import numpy as np

from llm import embed

# In the following steps, we 
# 1) Flatten all the codebooks into a single list of codes
# 2) The codes are embedded into a "semantic space" 
# 3) The embedded codes are clustered (put into groups) and the smallest clusters are discarded. 
# 4) The codes in a cluster are combined so that each cluster .
# This process allows us to find the most reliable codes of all texts. Of course, this has a trade-off and manual inspection is recommended.

# First, flatten the codebooks into a single list of codes.
codes = []
for name, textvalues in codelists:
    for iteration in textvalues:
        for codedict in iteration:
            codes.append(codedict['code'].replace("*", ""))

# Embed each code into the "semantic space".
vectors = []
for code in codes:
    result = embed(code)
    vectors.append(result['embedding'])

print(f"{len(vectors)} codes embedded.")

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
        html += ", ".join(items)
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

# Create figure for the plots
fig, axs = plt.subplots(2, 1, figsize=(15,10))
ax1, ax2 = axs

# This is called everytime we change the slider
def update_plot(threshold):

    # First update the tree-plot
    ax1.clear()
    dendrogram(Z, ax=ax1, no_labels=True)
    ax1.axhline(y=threshold, color='r', linestyle='--')

    # Then update the WCSS plot
    ax2.clear()
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
    display(fig)
    plt.close()

# Finall, create and show interactive clustering widget.
interactive_plot = widgets.interactive(update_plot, threshold=threshold_slider)
display(widgets.VBox([
    threshold_slider,
    cluster_info,
    interactive_plot.children[-1],
    cluster_contents
]))
update_plot(default_threshold)

# %%
# Here we actually apply the threshold and create the clusters.

# The threshold-parameter for the clustering (bigger threshold means bigger clusters. Here, smaller values are probably better.)
threshold = 0.25

# Discard clusters with less than {min_size} elements.
min_size = 2

# Compute the clusters with the selected threshold
distance_matrix = pdist(vectors, metric='cosine')
Z = linkage(distance_matrix, method='average')
clusters = fcluster(Z, threshold, criterion='distance')

# Reorganize to a simple list of clusters with smallest clusters (size < {min_size}) discarded
cluster_dict = {}
for cluster_id, code in zip(clusters, codes):
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []
    cluster_dict[cluster_id].append(code)
code_clusters = sorted([cluster[1] for cluster in cluster_dict.items() if len(cluster[1]) >= min_size], key=lambda x: len(x))

# Finally print the resulting clusters to check
for idx, cluster in enumerate(code_clusters):
    print(f"{idx+1}: {cluster}")


# %%
from llm import generate_simple

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
    Olet laadullisen tutkimuksen avustaja. Saat syötteenä klusteroinnin seurauksena syntyneen klusterin koodit. 
    Tarkoituksesi on yhdistää annettu lista koodeja yhdeksi kuvaavaksi koodiksi kenties vain valitsemalla yksi tai sitten muuten yhdistämällä. 
    Vastauksen pitäisi olla mahdollisimman yksinkertainen.
    """

    # For data, we set a comma-separated list of cluster elements.
    data = ", ".join(cluster)

    # Send the request to llm.
    result = generate_simple(instruction, data, seed=10, n_context=1024, output_format=output_format)

    # Extract the code
    code = json.loads(result['message']['content'])

    # And store it
    final_codes.append(code)

# Print the final codebook.
print("Final codes:")
for idx, code in enumerate(final_codes):
    print(f"{idx+1}: {code} ({', '.join(code_clusters[idx])})")

print("\nIn a format for easy copying:")
print(list(dict.fromkeys(reversed([codedict['code'] for codedict in final_codes]))))

# %%
