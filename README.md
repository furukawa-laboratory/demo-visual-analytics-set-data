# GenerativeManifoldModelingNetwork
<img width="1438" alt="Viewer" src="https://user-images.githubusercontent.com/19607449/92686237-38a03200-f374-11ea-8d6c-1dd67ed7ff28.png">
In the paper, *Multi-Aspect Visualization for Team Formation Support Based on Generative Manifold Modeling Network*, we introduce the visualization method to support forming the team.
It's the network consisting of *generative manifold modelings*.
Proposed method can visualize the relationship among own team's composition, opposing team's one and own team's performance.
The purpose of this repository is sharing our implementation of proposed method and providing the demonstration of interactive visualization.

## Getting started
Please build the Python environment satisfies the requirements in the `pyproject.toml`.
If you use poetry, run `poetry install`.
In addition, please make `tkinter` available in the Python environment because the visualization implemented by matplotlib works only when backend is set to `TkAgg`.

## Demonstration of interactive visualization
This demonstration provides a GUI Interface that facilitates knowledge discovery by clicking topological maps, as shown in the image above.
We use the data of 1228 games held in 2018–2019, obtained from [Basketball Reference website](https://www.basketball-reference.com/).

By running `learn_and_visualize.py`, you can try a prototype of the proposed method's interactive visualization.
`learn_and_visualize.py` both trains and visualizes the model, but since the trained model is saved in `dumped`, the visualization is done immediately.
If you want to apply the model to the desired other dataset, code a script to learn model by referring to `learn_and_visualize.py`.

## How to use?
In preparation...

## Contact us
If you have any questions or requests, please give us an Issue!
