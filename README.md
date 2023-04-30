# Learning-Causal-Inference-using-NNs

The details of the project are given in the presentation: https://docs.google.com/presentation/d/161cAW0kjq6RxzMlWDM5duIk_IRJEUDYgLpVWDJ0AHeY/edit?usp=sharing

To train the visual model, just run the script `train_visual.py`

To train the audio models, first prepare the dataset by running the script `prepare_dataset.py` in dataset folder. This will create the spectrograms  for different positions.
Now you can use this dataset to train the audio model by running the script `python train_audio.py`. You can specify the type of the audio model (branch, concat, left_only, right_only) inside the train_visual script.
