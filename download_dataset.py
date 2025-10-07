import kagglehub

# Download latest version
path = kagglehub.dataset_download("ernestojaguilar/shortterm-electricity-load-forecasting-panama")

print("Path to dataset files:", path)