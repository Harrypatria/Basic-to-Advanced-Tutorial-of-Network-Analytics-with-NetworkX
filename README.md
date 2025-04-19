# 🌐 NetworkX: From Basics to Advanced Network Analytics

<div align="center">
  
![NetworkX Logo](https://networkx.org/_static/networkx_logo.svg)

[![GitHub stars](https://img.shields.io/github/stars/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX?style=social)](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX?style=social)](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX/network/members)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 🌟 Overview

This repository offers a comprehensive guide to mastering NetworkX, a powerful Python library for creating, manipulating, and studying the structure, dynamics, and functions of complex networks. Whether you're a data scientist, a researcher, or a developer interested in network analysis, this tutorial will take you from the basics to advanced techniques, complete with practical examples and industry best practices.

## ✨ Key Features

- **📚 Comprehensive Learning Path**: From basic graph creation to advanced network analysis.
- **📈 Real-world Applications**: Social network analysis, biological networks, transportation systems, and more.
- **📊 Data Science Integration**: Seamless integration with pandas, NumPy, and visualization tools.
- **📈 Interactive Notebooks**: Run all examples directly in Google Colab.
- **🛡️ Production-ready Code**: Best practices for robustness, scalability, and performance.
- **🌐 Web and Mobile Integration**: Techniques for deploying network analysis in web and mobile applications.

## 🚀 Quick Start

### Option 1: Run in Google Colab (No Installation Required)
<p align="left" style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://colab.research.google.com/" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Main Tutorial In Colab" />
  </a>
</p>

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX.git
cd Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX

# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter notebook
jupyter notebook
```

## 📋 Notebooks Included

1. **[NetworkX_Basics.ipynb](https://colab.research.google.com/drive/1sbuG_dWd465P7FrDGj4qZRsCfvkefgWI?usp=sharing)** - Core NetworkX functionality from basics to advanced techniques.
2. **[Social_Network_Analysis.ipynb](https://colab.research.google.com/drive/1fH3Cr47aQQCohORrOigvLHDb-FjXLjW6?usp=sharing)** - Analyzing social networks with NetworkX.
3. **[Biological_Networks.ipynb](https://colab.research.google.com/drive/1_MIxE1voCeeIq_ho2vUL1tSyIC-dBdfa?usp=sharing)** - Exploring biological networks using NetworkX.

## 📋 Table of Contents

1. **Fundamentals**
   - Creating Graphs
   - Adding Nodes and Edges
   - Basic Graph Operations

2. **Intermediate Operations**
   - Graph Algorithms
   - Network Analysis
   - Visualization Techniques

3. **Advanced Techniques**
   - Performance Optimization
   - Custom Graph Generators
   - Integration with Other Libraries

4. **Real-world Applications**
   - Social Network Analysis
   - Biological Networks
   - Transportation Systems

5. **Production Best Practices**
   - Scalability
   - Robustness
   - Deployment Strategies

## 🔥 Optimization Techniques Showcase

| Technique | Description | Performance Gain |
|-----------|-------------|-----------------|
| Graph Pruning | Removing unnecessary nodes and edges | Up to 20x faster operations |
| Efficient Data Structures | Using optimized data structures for storage | 5-10x faster access |
| Parallel Processing | Leveraging multi-core processors for computations | 2-4x faster processing |
| Caching Results | Storing intermediate results to avoid redundant calculations | 10-50x faster repeated queries |

## 💡 Real-world Example: Social Network Analysis

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add nodes and edges
G.add_nodes_from(["Alice", "Bob", "Charlie", "Diana"])
G.add_edges_from([("Alice", "Bob"), ("Alice", "Charlie"), ("Bob", "Diana"), ("Charlie", "Diana")])

# Analyze the graph
degree_centrality = nx.degree_centrality(G)
print("Degree Centrality:", degree_centrality)

# Visualize the graph
nx.draw(G, with_labels=True, node_color=list(degree_centrality.values()), node_size=500, cmap='viridis')
plt.show()
```

## 🔧 Advanced Use Cases

### Web Application Integration (Flask)

```python
from flask import Flask, jsonify
import networkx as nx

app = Flask(__name__)

@app.route('/api/network')
def get_network():
    G = nx.erdos_renyi_graph(100, 0.5)
    data = nx.node_link_data(G)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
```

### Data Science Pipeline

```python
import pandas as pd
import networkx as nx

# Load data
df = pd.read_csv('network_data.csv')

# Create graph from DataFrame
G = nx.from_pandas_edgelist(df, source='source', target='target')

# Analyze the graph
clustering = nx.clustering(G)
print("Clustering Coefficients:", clustering)
```

### Interactive Dashboard with Plotly Dash

```python
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.erdos_renyi_graph(100, 0.5)

# Initialize Dash app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Network Analysis Dashboard"),
    dcc.Graph(id='network-graph')
])

# Callbacks
@app.callback(
    Output('network-graph', 'figure'),
    [Input('network-graph', 'clickData')]
)
def update_graph(clickData):
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = px.scatter(x=edge_x, y=edge_y, mode='lines')
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = px.scatter(x=node_x, y=node_y, mode='markers')
    fig = px.scatter()
    fig.add_trace(edge_trace.data[0])
    fig.add_trace(node_trace.data[0])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

## 📚 Learning Path

This tutorial is designed to progressively build your NetworkX expertise:

1. **Day 1**: Basic graph creation and manipulation
2. **Day 2**: Adding nodes and edges, basic operations
3. **Day 3**: Graph algorithms and network analysis
4. **Day 4**: Visualization techniques
5. **Day 5**: Real-world applications
6. **Day 6**: Advanced techniques and optimization
7. **Day 7**: Deployment and integration

## 🤔 Why NetworkX?

- **Powerful and Flexible**: Supports a wide range of graph types and operations.
- **Easy to Use**: Simple API for creating and manipulating graphs.
- **Extensive Documentation**: Comprehensive guides and examples.
- **Community Support**: Active community and regular updates.
- **Integration**: Works seamlessly with other Python libraries like pandas, NumPy, and Matplotlib.

## 📊 Industry Applications

- **Social Networks**: Analyzing social connections and interactions.
- **Biological Networks**: Studying protein interactions and biological pathways.
- **Transportation Systems**: Optimizing routes and analyzing traffic flow.
- **Communication Networks**: Designing and analyzing communication systems.
- **Data Science**: Exploring and visualizing complex data relationships.

## 🛠️ Troubleshooting Common Issues

<details>
<summary><b>Graph Visualization Issues</b></summary>

When encountering issues with graph visualization:

```python
# Solution 1: Use a different layout
pos = nx.spring_layout(G)  # or nx.circular_layout(G), nx.random_layout(G), etc.

# Solution 2: Adjust node and edge sizes
nx.draw(G, pos, node_size=500, with_labels=True, edge_color='gray')
plt.show()
```
</details>

<details>
<summary><b>Performance Issues with Large Graphs</b></summary>

When working with large graphs:

```python
# Solution 1: Use efficient data structures
G = nx.Graph()

# Solution 2: Prune unnecessary nodes and edges
G.remove_nodes_from(list(nx.isolates(G)))
G.remove_edges_from(nx.selfloop_edges(G))
```
</details>

<details>
<summary><b>Memory Management for Large Graphs</b></summary>

When working with large graphs:

```python
# Solution 1: Use generators for large datasets
def load_large_graph(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(u, v)
    return G
```
</details>

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

<div align="center">

## 🌟 Support This Project
**Follow me on GitHub**: [![GitHub Follow](https://img.shields.io/github/followers/Harrypatria?style=social)](https://github.com/Harrypatria?tab=followers)
**Star this repository**: [![GitHub Star](https://img.shields.io/github/stars/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX?style=social)](https://github.com/Harrypatria/Basic-to-Advanced-Tutorial-of-Network-Analytics-with-NetworkX/stargazers)
**Connect on LinkedIn**: [![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harry-patria/)

Click the buttons above to show your support!

</div>
