# GenerativeManifoldModelingNetwork
![image](https://user-images.githubusercontent.com/19607449/122664638-8ab49480-d1dd-11eb-95ab-43c9d11099ea.png)

In the paper, *Visual analytics of set data for knowledge discovery and member selection support*([arXiv:2104.09231](https://arxiv.org/abs/2104.09231)), we introduce the visualization method to support forming the team.
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

### What functions are available in this demo?
All the features shown in the paper are available. See [the paper](https://arxiv.org/abs/2104.09231) for details.

## Contact us
If you have any questions or requests, please give us an Issue!
