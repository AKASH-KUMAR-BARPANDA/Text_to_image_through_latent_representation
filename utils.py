import kagglehub
import shutil

cache_path = kagglehub.dataset_download("adityajn105/flickr8k")
shutil.copytree(cache_path, "dataset", dirs_exist_ok=True)
print("Copied to:", "dataset")