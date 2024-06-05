# Molecule Generation using Diffusion Models on Graphs
The code for training models and generating new samples using trained models is found in the `Graph_Framework` folder.

## Folder structure for Graph_Framework
- `configs` is for configuration files for `main.py`, `generate.py`, and `process_dataset`.
- `datasets` is for python files for each dataset. 
- `diffusion` contains the implemented types of diffusion and noise schedules.
- `models` is for all the models used in the framework and must follow the specified structure below.
- `runs` contains the runs and contains a folder for each name specified in used config file.
- `utils` contains files with predefined helper functions used by the framework and to be used by any users.

## Requirements
Setup requirements that must be followed to use the framework.

### Adding a model to the framework
For an example see the `models/modelname` folder.
- Create a folder with the name of the model in the `models` folder.
- The name of the folder must match the lower case name of the model class, e.g. `models/transformer` for a class called `Transformer`.
- Create these files in the folder:
    - `model.py`: the file with the model class
    - `train.py`: the file with methods called during training
    - `sample.py`: the file with methods called during generation
- Add the following methods in the `train.py` file:
    - `loss_fn(x_0, con_diffusion, cat_diffusion, model) -> loss`
    - `val_fn(val_dataset, con_diffusion, cat_diffusion, model, decode_atom, decode_bond, epoch, config) -> None`
- Add the following methods in the `sample.py` file:
    - `sample_batch(n_samples, dataset, model, device) -> x_T`
    - `sample_reverse(con_diffusion, cat_diffusion, model, t, x_t) -> x_t-1`
    - `sample_mols(x_0, dataset) -> mols`

### Adding a dataset to the framework
- Create a file in the `datasets` folder that contains a class for a PyTorch dataloader. 
- The name of the file must match the lower case name of the class, e.g. `dataset.py` and the class `Dataset`. 
- The dataset class must include two  accessable dictionaries, `decode_atom` and `decode_bond`, that are used to go from an integer to the correct atom or bond.

### Making a config file for training or generation
In the `configs` folder:
- See `template_train.yml` for a config file for training
- See `template_generate.yml` for a config file for generation
