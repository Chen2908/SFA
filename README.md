# SFA

This is the code for the SFA method presented in the paper Shapley-based Feature Augmentation.


### Datasets:

To download datasets use the following link:
[openml datasets](https://drive.google.com/uc?export=download&id=14_pDIR3H5BHbqytqxvRz7lQXtDjresg7)

Unzip the folder once downloaded.


-------------------------


### Packages reuirements:

To create an environment identical to the one we used to run the code:
conda create --name myenv --file spec-file.txt

-------------------------



### Run code:

XGBoost (run on GPU machine):

python main.py --dataset_id=**A number from 0-14 for binary, from 0-4 for multi class** --task=**binary or multi** --model_name=xgb --seed=**A number of your choice, we used 1-5** --compare=**True if you want to compare results to Featuretools and PCA Augment, otherwise or False**
For instance:
python main.py --dataset_id=3 --task=multi --model_name=xgb --seed=1 --compare=False


-----------------


LightGBM (run on CPU machine):

python main.py --dataset_id=**A number from 0-14 for binary, from 0-4 for multi class** --task=**binary or multi** --model_name=lgbm --seed=**A number of your choice, we used 1-5** --compare=**True if you want to compare results to Featuretools and PCA Augment, otherwise or False**
For instance:
python main.py --dataset_id=3 --task=multi --model_name=lgbm --seed=1 --compare=False

-----------------


Random forest (run on GPU machine):

python main.py --dataset_id=**A number from 0-14 for binary, from 0-4 for multi class** --task=**binary or multi** --model_name=random_forest --seed=**A number of your choice, we used 1-5** --compare=**True if you want to compare results to Featuretools and PCA Augment, otherwise or False**
For instance:
python main.py --dataset_id=3 --task=multi --model_name=random_forest --seed=1 --compare=False

-----------------

