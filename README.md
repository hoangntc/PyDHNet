# PyDHNet: An Open-Source Toolkit for Dynamic Heterogeneous Network Representation Learning


## Directory structure:

```
.
|   README.md
|   environment.yml
|
|--- dataset
|--- model
|--- ouput
|--- dedenpencies
|   |-- littleballoffur: module for graph sampling
|   |-- prepare_data: module for data preprocessing
|
|--- PyDHNet
|   |-- config
|   |   dblp.json
|   |-- src
|   |   data_preparation.py: module for data preparation
|   |   datasets.py: data module
|   |   model.py: model module
|   |   trainer.py: trainer module
|   |   inference.py: inference agent
|   |   evaluation.py: evaluation module
|   |   utils.py: utils functions
```


## Overall framework

![Overall framework](/figs/overview.png)


## Installation

### Libraries

To install all neccessary libraries, please run:

```bash
conda env create -f environment.yml
```

In case, the version of Pytorch and Cuda are not compatible on your machine, please remove all related lib in the `.yml` file; then install Pytorch and Pytorch Geometric separately. If you want to create an environment without using existing file, please refer to `installation.md` file. 


### PyTorch
Please follow Pytorch installation instruction in this [link](https://pytorch.org/get-started/locally/).


### Torch Geometric
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```
where `${TORCH}` and `${CUDA}` is version of Pytorch and Cuda.


## Experimental replication

### Dataset
You can run the demo with the DBLP dataset used in the paper, the data is in `dataset/dblp` folder.
To use your own dataset, you need to prepare 
    - One config file with json format named `data_name.json` and put in `PyDHNet/config`
    - Four input files put under the folder `dataset` with the name  `data_name`

    1. node_types.csv: format of each row `node_id (int), node_type (int), node_type_name (str)`

    ```
    node_id,node_type,node_type_name
    0,0,author
    1,0,author
    2,0,author
    3,0,author
    4,0,author
    ```
    2. temporal_edge_list.txt: format of each row `source_node_id (int), target_node_id (int), time_id (int)`

    ```
    1840 1 6
    1840 2 6
    1840 3 6
    1841 4 4
    1841 5 4
    ```

    3. temporal_subgraphs.pth: format of each row `subgraph_ids, time_id, label`

    ```
    1883-90-105-12693-12812-13117-13235-13273-13682-14027-14158-14241-14387-14517	0	uai	
    1884-105-121-12736-12827-13072-13329-14517	0	uai	
    1909-182-183-12636-12640-12749-12776-12782-12807-13039-13040-13124-13676-14308-14410-14489-14519	0	cikm	
    1930-242-243-13072-13228-13702-14073-14089-14311-14519	0	cikm	
    1972-346-347-12578-12693-12893-13437-13473-13595-13740-14421-14523	0	colt	
    ```

    4. data.pkl: a dictionary for train/val/test dataloader
    
    ```
    data = {0: {
    'node_id': 800,
    'subgraph_idx': {0: [1, 2], 1: [4, 10], 2: [], 3: [8], 4: [99, 100, 101], 5: [7]},
    'label': 'kdd',
    'dataset': 'train',
    'time_id': 3,
    },
    }
    ```
-     

### Usage

#### Subgraph Sampler

```python
from PyDHNet.subgraph_sampler import TemporalSubgraphSampler

sampler = TemporalSubgraphSampler(
    node_path,
    edge_path, 
    sampled_node_ids, 
    max_size=5, 
    number_of_nodes=20,
    seed=0,
    output_dir='./',
)

sampler.sampling_temporal_subgraph()
sampler.write_temporal_subgraphs()

```
#### Network Representation Learning

```python
from PyDHNet import PyDHNet

# Instance initialization
pydhnet = PyDHNet(config_path='./PyDHNet/config/dblp.json')

# or
config_dict = {
    'name': 'dblp',
    'num_time_steps': 8,
    'batch_size': 64,
    'learning_rate': 0.0001,
    'checkpoint_dir': './model',
}
pydhnet = PyDHNet(config_dict=config_dict)

#-----------------------------------------
# Manual running
# 1. Data preprocessing
pydhnet.preprocess_data()

# 2. DataModule, ModelModule, Trainer initialization
data_module, model_module, trainer = pydhnet.initialize()
pydhnet.train(data_module, model_module, trainer)

# 3. Embedding generating
restore_model_dir = str(pydhnet.config['checkpoint_dir'])
restore_model_name = 'name.ckpt'
output_dir = str(PROJ_PATH / 'output')
pydhnet.generate_embedding(
    data_module, model_module, restore_model_dir, restore_model_name, output_dir)

#-----------------------------------------
# Full pipeline running
pydhnet.run_pipeline()    
```

#### Evaluation

```python
from PyDHNet.evaluation import (
    eval_link_prediction, 
    eval_node_classification
)

lp_result = eval_link_prediction(
    source_features, 
    target_features, 
    labels, 
    train_val_test_index
)

nc_result = eval_node_classification(
    features, 
    labels, 
    train_val_test_index
)
```
