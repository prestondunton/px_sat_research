from sat import read_sat_problem

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config


def scatter_3d(data, metrics=None):

    columns = st.columns(4)
    with columns[0]:
        x_col = st.selectbox(label='X', options=data.columns)
    with columns[1]:
        y_col = st.selectbox(label='Y', options=data.columns)
    with columns[2]:
        z_col = st.selectbox(label='Z', options=data.columns)
    with columns[3]:
        color_col = st.selectbox(label='Color', options=data.columns.tolist())

    data['size'] = 1
    fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, size='size', size_max=10, color=color_col,
    #                    range_z=(0,15),
                    #    range_color=(metrics[color_col]['min'], metrics[color_col]['max']),
                        hover_data={'problem': True,
                                    'trial': True,
                                    x_col: True,
                                    y_col: True,
                                    z_col: True,
                                    'size': False
                                    })
    st.plotly_chart(fig, use_container_width=True)


def prob_dist():
    #s = np.linspace(0, 2 * np.pi, 240)
    #t = np.linspace(0, np.pi, 240)

    s = np.linspace(-5, 5, 100)
    t = np.linspace(-5, 5, 100)

    tGrid, sGrid = np.meshgrid(s, t)

    st.write(tGrid)
    st.write(sGrid)

    #r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
    #x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
    #y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
    #z = r * np.cos(tGrid)  # z = r*cos(t)

    x = tGrid
    y = sGrid
    z = log_normal(x, 0, 1)


    surface = go.Surface(x=x, y=y, z=z)
    data = [surface]

    layout = go.Layout(
        title='Parametric Plot',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)


def normal(x, mu, s2):
    return (1 / math.sqrt(2 * math.pi * s2)) * math.exp(-0.5 * math.pow((x-mu) / s2, 2))

def log_normal(x, mu, s2):
    return (1 / (x * math.sqrt(2 * math.pi * s2))) * math.exp(-0.5 * math.pow(math.log(x-mu), 2) / s2)


def scatter_plots():

    data = pd.read_csv('./Honors Option/results/graph_decomposition.csv', index_col=0)
    data = data[['trial', 'problem', 'm', 'n', 'm / n', 'm\'', 'n\'',  'm / m\'','dq', 'dq\'', 'dq / dq\'']]
    for column in data.columns:
        if data[column].dtype == 'float64':
            data[f'log({column})'] = np.log(data[column])

    metrics = data.describe()

    if st.button('resample'):
        problems = data['problem'].unique()
        np.random.shuffle(problems)
        problems = problems[0:30]
        data = data[data['problem'].isin(problems)]
        #data.sort_values(by=['m / n'], inplace=True)

    st.dataframe(data)

    scatter_3d(data, metrics)

    #prob_dist()


def graphs():

    sat = read_sat_problem('./toy_sat_problems/toy_sat1.cnf')

    edges = []
    nodes = []
    node_size = 750

    for i in range(sat.m):
        if len(sat.clauses[i]) == 1:
            nodes.append(Node(id=abs(list(sat.clauses[i])[0]), size=node_size, renderLabel=True))

        else:
            # K-bounded.  For example, in MAX-K-SAT, loops j and k together will only run (K choose 2) times
            for variable_1 in sat.clauses[i]:
                for variable_2 in sat.clauses[i]:
                    if variable_1 < variable_2:
                        nodes.append(Node(id=abs(variable_1), size=node_size, renderLabel=True))
                        nodes.append(Node(id=abs(variable_2), size=node_size, renderLabel=True))
                        edges.append(Edge(source=abs(variable_1), target=abs(variable_2)))

    config = Config(width=800,
                    height=800,
                    directed=False,
                    collapsible=False,
                    panAndZoom=False,
                    staticGraph=False)
    agraph(nodes=nodes, edges=edges, config=config)


def main():
    graphs()


if __name__ == '__main__':
    main()