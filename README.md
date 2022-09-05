# plantnet

## Getting started
- original dataset can be found here: (https://zenodo.org/record/5645731)
- Please serve the images in './images'. Subfolders: images_test and images_train
- Inside 'images_test' and 'images_train' are folders for every species, containing the corresponding images.
- In Anaconda, run: conda env create -f plantnet_300k_env.yml
- As (pre-trained)-model choose either ResNet50, ResNext101 or VisionTransformer (the models are imported and hardcoded)
- For training: Adjust batch_size and n_epochs in main.py according to inline comments
- Run main.py to train and test the (pre-trained)-model
- Output of main.py: Weights of best model and csv-file with results over each epoch
