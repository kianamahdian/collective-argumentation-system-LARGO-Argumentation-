import streamlit as st
import json
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
import base64
import altair as alt
import numpy as np
from aggregators import compute_aggregations


with open('dataset_with_relations.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

arguments = dataset['arguments']
original_agents = dataset['agents']
arg_ids = [arg['id'] for arg in arguments]
arg_by_id = {arg['id']: arg for arg in arguments}


if 'simulated_agents' not in st.session_state:
    st.session_state['simulated_agents'] = []
if 'current_agents' not in st.session_state:
    st.session_state['current_agents'] = original_agents.copy()
if 'previous_labels' not in st.session_state:
    st.session_state['previous_labels'] = {agent['id']: agent['labels'].copy() for agent in original_agents}


attackers_of = defaultdict(set)
defenders_of = defaultdict(set)

for arg in arguments:
    src = arg['id']
    for target, rel in arg.get('relationships', {}).items():
        if rel == 'attack':
            attackers_of[target].add(src)
        elif rel == 'defend':
            defenders_of[target].add(src)


def compute_results(agents_override=None):
    temp_agents = agents_override or st.session_state['current_agents']
    return compute_aggregations(temp_agents, arguments, arg_ids, attackers_of, defenders_of)


results, final_decision, in_f, out_f, undec_f, current_in_counts, current_out_counts, current_undec_counts, mild_status, behavior = compute_results()


st.set_page_config(page_title="Collective Argumentation Dashboard", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; color: #1f77b4; }
    .section-header { color: #ff7f0e; font-size: 20px; }
    .label-in { background-color: #2ca02c; color: white; padding: 2px 6px; border-radius: 4px; }
    .label-out { background-color: #d62728; color: white; padding: 2px 6px; border-radius: 4px; }
    .label-undec { background-color: #7f7f7f; color: white; padding: 2px 6px; border-radius: 4px; }
    .stButton > button { background-color: #17becf; color: white; }
    .stSidebar { background-color: #f0f2f6; }
    .changed-label { background-color: #ffeb3b; }  /* Highlight changed labels */
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Controls & Simulations")

    # Use sorted keys for consistent ordering (matching paper table order if possible)
    method_options = sorted(results.keys())
    selected_method = st.selectbox("Highlight Method", method_options)

    st.subheader("Add Simulated Agent")
    sim_agent_id = st.text_input("New Agent ID", value=f"SimAgent{len(st.session_state['simulated_agents']) + 1}")
    sim_labels = {}
    for aid in arg_ids:
        sim_labels[aid] = st.selectbox(f"Label for {aid}", ['in', 'out', 'undec'], key=f"sim_{aid}_{len(st.session_state['simulated_agents'])}")
    if st.button("Add Agent"):
        new_agent = {"id": sim_agent_id, "labels": sim_labels}
        st.session_state['simulated_agents'].append(new_agent)
        st.session_state['current_agents'] = original_agents + st.session_state['simulated_agents']
        results, final_decision, in_f, out_f, undec_f, current_in_counts, current_out_counts, current_undec_counts, mild_status, behavior = compute_results(st.session_state['current_agents'])
        st.success(f"Agent '{sim_agent_id}' added!")
        st.rerun()  # Refresh to update highlight options

    if st.button("Clear Simulated Agents"):
        st.session_state['simulated_agents'] = []
        st.session_state['current_agents'] = original_agents.copy()
        results, final_decision, in_f, out_f, undec_f, current_in_counts, current_out_counts, current_undec_counts, mild_status, behavior = compute_results()
        st.success("Simulated agents cleared!")
        st.rerun()

    results_json = json.dumps(results, indent=4)
    st.download_button("Download Results JSON", results_json, file_name="aggregation_results.json")


st.title("🗣️ Collective Argumentation Dashboard")
st.markdown(f'<p class="big-font">Goal: {dataset["goal"]}</p>', unsafe_allow_html=True)


col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="section-header">Arguments</p>', unsafe_allow_html=True)
    arg_df = pd.DataFrame([{'ID': arg['id'], 'Text': arg['text'], 'Group': arg.get('group', '')} for arg in arguments])
    st.dataframe(arg_df.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}))

with col2:
    st.markdown('<p class="section-header">Vote Distribution Charts</p>', unsafe_allow_html=True)
    vote_data = []
    for aid in arg_ids:
        vote_data.append({'Argument': aid, 'In': current_in_counts[aid], 'Out': current_out_counts[aid],
                          'Undec': current_undec_counts[aid]})
    vote_df = pd.DataFrame(vote_data)
    chart = alt.Chart(vote_df.melt('Argument')).mark_bar().encode(
        x='Argument', y='value', color='variable', tooltip=['Argument', 'variable', 'value']
    ).properties(width=300, height=200)
    st.altair_chart(chart)


st.markdown('<p class="section-header">Agents & Labels</p>', unsafe_allow_html=True)
agent_data = []
for agent in st.session_state['current_agents']:
    row = {'Agent ID': agent['id']}
    is_new = agent['id'].startswith('SimAgent')
    prev_labels = st.session_state['previous_labels'].get(agent['id'], {})
    for aid, label in agent['labels'].items():
        color_class = 'label-in' if label == 'in' else 'label-out' if label == 'out' else 'label-undec'
        if is_new or label != prev_labels.get(aid):
            row[aid] = f'<span class="{color_class} changed-label">{label}</span>'  # Highlight new/changed
        else:
            row[aid] = f'<span class="{color_class}">{label}</span>'
    agent_data.append(row)
agent_df = pd.DataFrame(agent_data)
st.markdown(agent_df.to_html(escape=False), unsafe_allow_html=True)


st.session_state['previous_labels'] = {agent['id']: agent['labels'].copy() for agent in st.session_state['current_agents']}


st.markdown('<p class="section-header">Argumentation Graph</p>', unsafe_allow_html=True)
G = nx.DiGraph()
for arg in arguments:
    G.add_node(arg['id'], label=arg['id'] + ": " + arg['text'][:20] + "...")
for src_arg in arguments:
    for target, rel in src_arg.get('relationships', {}).items():
        dash = 'solid' if rel == 'attack' else 'dot'  # Solid for attack, dotted for defend
        G.add_edge(src_arg['id'], target, dash=dash, relation=rel)

pos = nx.spring_layout(G, seed=42)
edge_traces = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                  line=dict(width=2, color='black', dash=edge[2]['dash']),
                                  hoverinfo='text', text=f"{edge[0]} {edge[2]['relation']} {edge[1]}",
                                  mode='lines'))

node_trace = go.Scatter(x=[pos[node][0] for node in G.nodes()],
                        y=[pos[node][1] for node in G.nodes()],
                        text=[G.nodes[node]['label'] for node in G.nodes()],
                        mode='markers+text',
                        marker=dict(size=25, color='#1f77b4', line=dict(width=2, color='DarkSlateGrey')),
                        hoverinfo='text')

fig = go.Figure(data=edge_traces + [node_trace],
                layout=go.Layout(showlegend=False, hovermode='closest', title='Interactive Argumentation Graph'))
st.plotly_chart(fig, use_container_width=True)

# --- Aggregation Results ---
st.markdown('<p class="section-header">Aggregation Results</p>', unsafe_allow_html=True)
st.markdown(
    f'### Final Decision for Norm (N): <span class="label-{final_decision}">{final_decision.upper()}</span> (In: {in_f}, Out: {out_f}, Undec: {undec_f})',
    unsafe_allow_html=True)

# Sort the dataframe to match paper table order (approximate, based on typical grouping)
preferred_order = [
    'Opinion-First (OF)', 'Support-First (SF)', 'Balanced (BF)',
    'ABORDA_S', 'ABORDA_SDI', 'ABORDA_P', 'ABORDA_PDI',
    'ACOP_D', 'ACOP_DI(Att/Def)', 'ACOP_DI(Pro/Con)',
    'ARGVET_D', 'ARGVET_DI',
    'ACUMUL',
    'AKEMEN_D', 'AKEMEN_DI',
    'ASIMP_D', 'ASIMP_DI',
    'APREF_MLD', 'APREF_MD', 'APREF_MLD(T)', 'APREF_MD(T)', 'APREF_DIMLD', 'APREF_DIMD'
]
results_df = pd.DataFrame(results).T
results_df['Method'] = results_df.index
results_df['Mild'] = [mild_status.get(method, 'N/A') for method in results_df['Method']]
results_df['Behavior'] = [behavior.get(method, 'N/A') for method in results_df['Method']]
results_df = results_df.set_index('Method').reindex(preferred_order).reset_index()

highlight = results_df.style.apply(
    lambda row: ['background-color: #ffeb3b' if row.Method == selected_method else '' for _ in row], axis=1)
st.dataframe(highlight, use_container_width=True)


st.markdown('<p class="section-header">Related Paper PDF</p>', unsafe_allow_html=True)


def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


display_pdf('26_dec_Karimi_Collective_Argumentation_Based_on_Social_Choice_Methods.pdf')


if st.button("Run Random Simulation"):
    prev_labels = {agent['id']: agent['labels'].copy() for agent in st.session_state['current_agents']}
    for agent in st.session_state['current_agents']:
        for aid in arg_ids:
            agent['labels'][aid] = np.random.choice(['in', 'out', 'undec'])
    results, final_decision, in_f, out_f, undec_f, current_in_counts, current_out_counts, current_undec_counts, mild_status, behavior = compute_results(st.session_state['current_agents'])

    st.session_state['previous_labels'] = prev_labels
    st.info("Random simulation applied! Labels randomized; check highlighted changes in Agents table.")
    st.rerun()