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
# # Yleiset kysymykset
#
# Esitetään sama kysymys jokaiselle tekstinäytteelle ja tehdään lopuksi vielä kooste.
#
# Aloitetaan lukemalla näytteet tiedostojärjestelmä muistiin, ja printataan ne vielä esiin.

# %%
from utils import read_files
from utils import strip_webvtt_to_plain_text

# Read the texts of interest from the file system
contents = read_files(folder="data/linnut-03", prefix="inputfile")

# Remove timestamps if present
contents = [(fname, strip_webvtt_to_plain_text(text)) for fname, text in contents]

# Print to check that texts are correctly read
print("Luetut tekstit:")
print(f"\n--------------------------\n")
for fname, text in contents:
    print(f"{fname}:\n")
    print(f"{text}\n")
    print(f"\n--------------------------\n")

# %% [markdown]
# Esitetään seuraavaksi sama alla määritelty kysymys jokaiselle tekstille erikseen ja kerätään tulokset.

# %%
from llm import generate_simple

# Define the question
question = """
Esiintyykö tekstissä ristiriitaisia tunteita?
"""

# In a loop, ask the question for all the texts.

results = []
for fname, text in contents:
    # Define the instructions. 
    instruction = """
    Olet laadullisen tutkimuksen avustaja. Saat tekstinäytteen sekä siihen liittyvän kysymyksen. Lue teksti huolella ja vastaa ainoastaan sen perusteella kysymykseen.
    """

    # Define the content snippet given to the llm.
    content = f"Tekstinäyte: {text} \n\n Kysymys: \n\n {question}"

    # Generate the answer
    result = generate_simple(instruction, content, seed=10)

    # Extract the result
    answer = result
            
    # Store it
    results.append({
        "fname": fname,
        "question": question,
        "text": text,
        "answer": answer
    })

print("Vastaus kullekin tekstille:\n")
for result in results:
    print(f"{result['fname']}:\n\n {result['answer']}\n\n")

# %% [markdown]
# Lopuksi, pyydetään tekoälyltä tiivistelmä.

# %%
# Define the instructions. 
instruction = """
Olet laadullisen tutkimuksen avustaja. Sinulle annetaan syötteenä kysymys, joka on esitetty kokoelmalle tekstejä, jokaiselle tekstille erikseen, 
ja tekstikohtaiset vastaukset tähän kysymykseen. Luo näistä mahdollisimman havainnollistava tiivistelmä. Jos mahdollista, erittele esiintyminen teksteissä nimittäin.
"""

content = f"Kysymys: {question}"
content += "\n--------------------------------\n"
for result in results:
    content += f"Tiedostonnimi: {result['fname']}\n\n"
    content += f"Vastaus: \n\n {result['answer']}"
    content += "\n--------------------------------\n"

print("Syötteenä tekoälylle menee:\n\n")
print(content)

# Generate the answer
result = generate_simple(instruction, content, seed=10)

# Extract the result
answer = result

print("Vastaukseksi tekoälyltä tulee:\n\n")
print(answer)

# %%
