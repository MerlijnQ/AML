# AML
### Authors:
Merlijn Mulder\
Hinke Bolt\
Luca Colli\
Martijn Schippers

# Organization of files
The following categories of files can be found:
- **dataset** (map): the csv and excel files for importing the data
- **dataloader** (map): related to preprocessing data and initializing dataloaders
- **model** (map): all the details about the model
- **shap_explainer** (file): the class for applying SHAP
- **pipeline** (file): the pipeline for initializing dataloader, training a model and discarding a feature with SHAP
- **data** (map): json data of a running the pipeline
- **figure_pipeline** (map): code and figures of data preprocessing, heatmaps, RMSE, uncertainties