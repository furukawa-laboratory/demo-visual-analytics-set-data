# Demo of Visual Analytics system for set data
![Display of the demo](https://user-images.githubusercontent.com/19607449/126859486-396e6749-51c0-4955-80e0-bb8c99b7d09a.png)

In the paper, *Visual analytics of set data for knowledge discovery and member selection support*, we introduce the general method to build visual analytics system of set data.
This repository is to share our implementation of the prototype system build by proposed method in the paper and provide the demo of interactive visualization.


More details about our proposed method are in
- [Published version in Decision Support Systems](https://doi.org/10.1016/j.dss.2021.11...)
- [Accepted manuscript in arxiv](https://arxiv.org/abs/2104.09231)
- [How to use this demo after running in your environment (youtube)](https://www.youtube.com/watch?v=p0hF5OhX5DU)

## Getting started
Please build the Python environment satisfies the requirements in the `pyproject.toml`.
If you use poetry, run `poetry install`.

This demonstration provides a GUI Interface that facilitates knowledge discovery and member selection by clicking topological maps, as shown in the image above.
We use the data of 1228 games held in 2018–2019, obtained from [Basketball Reference website](https://www.basketball-reference.com/).

By running `learn_and_visualize.py`, you can try a prototype of the proposed method's interactive visualization.
`learn_and_visualize.py` both trains and visualizes the model, but since the trained model is saved in `dumped`, the visualization is done immediately.
If you want to apply the model to the desired other dataset, code a script to learn model by referring to `learn_and_visualize.py`.

### How to use? What features are available?
Please watch [this video](https://youtu.be/p0hF5OhX5DU) as a reference

## Contact us
If you have any questions or requests, please give us an Issue!
