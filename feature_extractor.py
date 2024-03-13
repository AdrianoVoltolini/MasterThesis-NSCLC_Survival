import numpy as np
import radiomics
import yaml
import pandas as pd
import SimpleITK as sitk
import os

spacings = pd.read_csv("xyz.csv", index_col=0)

with open("Params.yaml", "rb") as file:
    params = yaml.safe_load(file)

final_df = pd.DataFrame()

for name in os.listdir("new_CT"):

    image = np.load(f"new_CT/{name}")
    label = np.load(f"new_seg/{name}")

    name = name[:-4]

    x,y,z = spacings.loc[name,:]

    itk_image = sitk.GetImageFromArray(image)
    itk_image.SetOrigin((0,0,0))
    itk_image.SetSpacing((x,y,z))

    itk_label = sitk.GetImageFromArray(label)
    itk_label.SetOrigin((0,0,0))
    itk_label.SetSpacing((x,y,z))

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params)

    results = extractor.execute(itk_image,itk_label)

    pd_results = pd.DataFrame([results], index=[name])

    final_df = pd.concat([final_df, pd_results])
    print(name, "finished")

    final_df.to_csv("radiomic_features.csv")




