from gensim.models import KeyedVectors
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


st.set_page_config(
    page_title="MOPS vectors explorations",  # Title that appears on the browser tab
    page_icon="📊",  # Change the favicon (can be an emoji or a file path)
    layout="wide"  # Optional: can be "centered" or "wide"
)

# page description



st.markdown("""

# MOPS

This web app serves for interactive visualizations of a set of word embedding models trained on textual data from 25,000 articles covering 120 years of Pauline scholarship" 
""")

# loading data
# Load the dataset of positions of all words in 3D space
path = "/home/jupyter-vojta/notebooks/mops/data/"
with open(path + "vectors_positions3d_dict.pickle", "rb") as f:
    coordinates3d_dict = pickle.load(f)

with open(path + "vectors_positions_dict.pickle", "rb") as f:
    coordinates_dict = pickle.load(f)


filtered_vocab_df = pd.read_pickle(path + "bidecades_vocabs_counts_df.pickle")
filtered_vocab_df.index.name = 'word'

options = ['1900-1919', '1920-1939', '1940-1959', '1960-1979', '1980-1999', '2000-2019']
vectors_dict = {}
for bidecade in options:
    vectors_dict[bidecade] = KeyedVectors.load(path + "large_files/vectors_{0}_sample{1}.kv".format(bidecade, str(1000000)))

# streamlit interactive elements for data selection
subcorpus = st.selectbox('What subcorpus are you interested in:',
                         options,
                         index=options.index('1980-1999'))
target = st.text_input('What word are you interested in:', value='paul')
topn = st.slider('Select the number of nearest neighbours:', min_value=5, max_value=100, value=20)

def cosine_similarity_matrix(kv, wordlist):
    vectors = np.array([kv[word] for word in wordlist])
    # Compute the cosine similarity matrix (normalize and use dot product)
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    # Convert to a pandas DataFrame with words as both row and column labels
    similarity_df = pd.DataFrame(similarity_matrix, index=wordlist, columns=wordlist)
    return similarity_df

def plot_similarity_matrix(cosine_sim_matrix, title="Pairwise Vector Similarity"):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cosine_sim_matrix, cmap='Greens')
    fig.colorbar(cax, label='Cosine Similarity')
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Words", fontsize=12)
    ticks = np.arange(len(cosine_sim_matrix.columns))
    ax.set_xticks(ticks)
    ax.set_xticklabels(cosine_sim_matrix.columns, rotation=90, fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(cosine_sim_matrix.index, fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    return fig

def plot_gen(subcorpus, target, topn, coordinates3d_dict, filtered_vocab_df, vectors_dict):
    # selecting data based on subcorpus & target
    xs, ys, zs, words = coordinates3d_dict[subcorpus]

    word_dict = filtered_vocab_df.apply(
        lambda row: "<br>wordcount: " + (
            str(row[subcorpus]) if subcorpus in filtered_vocab_df.columns else "NA"),
            axis=1).to_dict()
    kv = vectors_dict[subcorpus]
    nns_tuples = kv.most_similar(target, topn=topn)
    wordlist = [target] + [tup[0] for tup in nns_tuples]
    sim_scores = [str(1)] + [str(np.round(tup[1], 2)) for tup in nns_tuples]

    # Normalize case for compatibility
    words = [word.lower() for word in words]
    wordlist = [word.lower() for word in wordlist]

    # Ensure words are matched in the correct order
    idx = [words.index(word) for word in wordlist if word in words]

    # Debugging: Ensure matching is correct
    #print("Wordlist:", wordlist)
    #print("Selected Indices:", idx)
    #print("Words in Plot:", [words[i] for i in idx])

    # Index the coordinates
    wordlist_xs, wordlist_ys, wordlist_zs = xs[idx], ys[idx], zs[idx]
    colors = ["darkred"] + ["black"] * len(nns_tuples)
    fontsizes = [18] + [14] * len(nns_tuples)

    hover_text = []
    for word, sim in zip(wordlist, sim_scores):
        if word in word_dict.keys():
            hover_text.append(word + word_dict[word] + "<br>similarity to target ({0}): {1}".format(target, sim))
        else:
            hover_text.append(
                word + "<br>wordcount: NA<br>similarity to target ({0}): {1}".format(target, sim))

    # define the figure object
    fig = go.Figure(data=go.Scatter3d(
        x=wordlist_xs,
        y=wordlist_ys,
        z=wordlist_zs,
        mode='markers+text',
        marker=dict(
            size=8,  # Enlarged marker size
            color='purple',
            opacity=0# .5  # Slightly higher opacity for markers
        ),
        text=wordlist,
        hovertext=hover_text,
        hoverinfo='text',
        textposition='top center',
        textfont=dict(
            size=fontsizes,  # Increased font size for readability
            color=colors,
            family='Arial, bold'  # Optional: bold or other readable fonts
        )
    ))

    fig.update_layout(
        title='Embeddings',
        scene=dict(
            xaxis=dict(title='', showgrid=False, showline=False, showticklabels=False, zeroline=False,
                        showbackground=False,
                        linecolor='rgba(0,0,0,0)'),
            yaxis=dict(title='', showgrid=False, showline=False, showticklabels=False, zeroline=False,
                        showbackground=False,
                        linecolor='rgba(0,0,0,0)'),
            zaxis=dict(title='', showgrid=False, showline=False, showticklabels=False, zeroline=False,
                        showbackground=False,
                        linecolor='rgba(0,0,0,0)'),
            bgcolor='rgba(255,255,255,0)'
        ),
        paper_bgcolor='rgba(255,255,255,255)',  # set the color of the area around the axes
        plot_bgcolor='rgba(255,255,255,255)',  # set the color of the entire chart
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=0),
        hovermode='closest',
        showlegend=False,
        uniformtext_minsize=12,
        uniformtext_mode='hide'  # Force minimum text size without displaying all labels
    )
    similarity_df = cosine_similarity_matrix(kv, wordlist[:35])
    nns_df = pd.DataFrame(nns_tuples, columns=["word", "similarity"]).set_index("word")
    fig_sim = plot_similarity_matrix(similarity_df, title="Pairwise Vector Similarity Matrix")
    return fig, nns_df, fig_sim

# Show the plot
try:
    fig, nns_df, fig_sim = plot_gen(subcorpus, target, topn, coordinates3d_dict, filtered_vocab_df, vectors_dict)
    st.markdown(f'''
    ### 3D projection of the {topn} words with the most similar vector to *{target}*
    ''')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'''
    ### {topn} words with the most similar vector to *{target}* with their similarity score
    ''')
    st.dataframe(nns_df, width=300)
    st.pyplot(fig_sim)
except:
    st.write(f"The word **{target}** is not in our corpus!")

# Initialize default groups if not in session state
if 'word_groups' not in st.session_state:
    st.session_state.word_groups = {
        'group0': ["paul", "saint", "apostle"],
        'group1': ["christianity", "faith", "theology", "theological", "doctrine",
                   "body", "spirit", "soul", "belief", "sacrifice", "righteousness",
                   "justification"],
        'group2': ["society", "social", "tradition", "traditional", "historical",
                   "community", "religious", "religion", "science", "critical",
                   "model", "identity", "economic", "explanation", "scientific"],
        'group3': ["jew", "law", "jewish", "israel", "hebrew", "jerusalem",
                   "judaism", "moses", "abraham"]
    }

# Create expander for word group editing
with st.expander("Edit Word Groups"):
    for group_name in ['group0', 'group1', 'group2', 'group3']:
        st.subheader(f"{group_name}")
        # Convert list to string for text area
        current_words = "\n".join(st.session_state.word_groups[group_name])
        new_words = st.text_area(
            "Edit words (one per line)",
            value=current_words,
            key=f"edit_{group_name}"
        )
        # Update group when changed
        st.session_state.word_groups[group_name] = [
            word.strip()
            for word in new_words.split('\n')
            if word.strip()
        ]

# Update the original variables for the plotting function
group0 = st.session_state.word_groups['group0']
group1 = st.session_state.word_groups['group1']
group2 = st.session_state.word_groups['group2']
group3 = st.session_state.word_groups['group3']

def draw_special_words(wordlist, color, xs, ys, words, ax, fontsize=6):
    color_idx = [word[0] for word in enumerate(words) if word[1] in wordlist]
    for x, y, w in zip(xs[color_idx], ys[color_idx] - 0.003, words[color_idx]):
        ax.text(x, y, w, horizontalalignment='center', fontsize=fontsize, color=color, zorder=3)

def plot_embeddings(subcorpus):
    keyed_vectors = vectors_dict[subcorpus]
    xs, ys, words = coordinates_dict[subcorpus]
    special_words = group0 + group1 + group2 + group3
    # all words as dots if not in lists
    dots_idx = [word[0] for word in enumerate(words) if word[1] not in special_words]
    fig, ax = plt.subplots() #figsize=(10, 10), dpi=300)
    ax.scatter(xs[dots_idx], ys[dots_idx], s=1, color="lightgrey", zorder=1)
    draw_special_words(group0, "green", xs, ys, words, ax=ax)
    draw_special_words(group1, "red", xs, ys, words, ax=ax)
    draw_special_words(group2, "blue", xs, ys, words, ax=ax)
    draw_special_words(group3, "purple", xs, ys, words, ax=ax)
    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)
    ax.set_axis_off()
    #plt.show()
    #fig.savefig("../figures/embeddings_{}.png".format(sub))
    return fig

st.pyplot(plot_embeddings(subcorpus))