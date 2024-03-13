import os
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd
from datetime import datetime, timedelta

class My_Dataset(Dataset):
  
  def __init__(self, args):

    self.args = args

    self.data_root = args.data_root
    self.additional_folder = args.additional_folder

    self.clinical_raw = pd.read_csv("ClinicalData.csv")
    self.clinical_data = self.clinical_edit(self.clinical_raw)

    if self.args.rna == True:
      self.RNA_raw = pd.read_csv("RNAseq_norm.csv",index_col=0)
      self.RNA_low_variance_mask = self.RNA_raw.T.var() > self.RNA_raw.T.var().quantile(0.05)

      self.RNA = self.RNA_raw[self.RNA_low_variance_mask].T
    
    if self.args.radiomic == True:
      self.radiomic_raw = pd.read_csv("radiomic_features.csv", index_col=0)
      self.radiomic_train_means = pd.read_csv("radiomic_train_means.csv", index_col=0).T
      radiomic_columns = [x for x in self.radiomic_raw.columns if x.startswith("diagnostics")==False]
      self.radiomic_data = self.radiomic_raw[radiomic_columns]
      

    self.ct_list = [x for x in os.listdir(self.data_root+"CT/") if x[:7] in list(self.clinical_data.index)]
    self.seg_list = [x for x in os.listdir(self.data_root+"seg/") if x[:7] in list(self.clinical_data.index)]

    self.additional_ct = [x for x in os.listdir(self.additional_folder+"CT/") if x[:7] in list(self.clinical_data.index)]
    self.additional_seg = [x for x in os.listdir(self.additional_folder+"seg/") if x[:7] in list(self.clinical_data.index)]

    self.ct_list.sort()
    self.seg_list.sort()
    self.additional_ct.sort()
    self.additional_seg.sort()

    self.x_train, x_temp, self.y_train, y_temp = train_test_split(self.ct_list,
                                                                            self.seg_list,
                                                                            test_size=args.test_size,
                                                                            random_state=args.random_state)


    self.x_train.extend(self.additional_ct)
    self.y_train.extend(self.additional_seg)

    self.x_val = x_temp[0:len(x_temp)//2]
    self.x_test = x_temp[len(x_temp)//2:]

    self.y_val = y_temp[0:len(y_temp)//2]
    self.y_test = y_temp[len(y_temp)//2:]

    if self.args.rna == True:
      self.x_train = [x for x in self.x_train if x[:7] in list(self.RNA.index)]
      self.x_val = [x for x in self.x_val if x[:7] in list(self.RNA.index)]
      self.x_test = [x for x in self.x_test if x[:7] in list(self.RNA.index)]

      self.y_train = [x for x in self.y_train if x[:7] in list(self.RNA.index)]
      self.y_val = [x for x in self.y_val if x[:7] in list(self.RNA.index)]
      self.y_test = [x for x in self.y_test if x[:7] in list(self.RNA.index)]
    
    if self.args.radiomic == True:
      self.x_train = [x for x in self.x_train if x[:7] in list(self.radiomic_data.index)]
      self.x_val = [x for x in self.x_val if x[:7] in list(self.radiomic_data.index)]
      self.x_test = [x for x in self.x_test if x[:7] in list(self.radiomic_data.index)]

      self.y_train = [x for x in self.y_train if x[:7] in list(self.radiomic_data.index)]
      self.y_val = [x for x in self.y_val if x[:7] in list(self.radiomic_data.index)]
      self.y_test = [x for x in self.y_test if x[:7] in list(self.radiomic_data.index)]

    self.input_d = args.input_D
    self.input_h = args.input_H
    self.input_w = args.input_W

    self.phase = args.phase

  def cambia_date(self, x):
    return datetime.strptime(x,"%m/%d/%Y")
  
  def clinical_edit(self, clinical_raw):
    survival_list = list()
    clinical_temp = clinical_raw.copy()

    for sample in clinical_raw.iterrows():

      date_last = self.cambia_date(sample[1]["Date of Last Known Alive"])
      date_CT = self.cambia_date(sample[1]["CT Date"])
      days_CT_surgery = timedelta(days=sample[1]["Days between CT and surgery"])

      if date_last - date_CT > days_CT_surgery: # caso in cui hanno fatto surgery dopo della CT
          delta = date_last - date_CT - days_CT_surgery
      else: # caso in cui hanno fatto surgery prima della CT
          delta = date_last - date_CT + days_CT_surgery

      survival_list.append(delta.days)
  

    clinical_temp["Days Survived"] = survival_list
    print(clinical_temp["Days Survived"].max())
    clinical_data = clinical_temp[clinical_temp["Pack Years"] != "Not Collected"]
    clinical_data = clinical_data[clinical_data["Pathological T stage"] != "Not Collected"]
    clinical_data = clinical_data[clinical_data["Pathological N stage"] != "Not Collected"]
    clinical_data = clinical_data[clinical_data["Pathological M stage"] != "Not Collected"]
    clinical_data = clinical_data[clinical_data["Weight (lbs)"] != "Not Collected"]
    clinical_data = clinical_data[clinical_data["Chemotherapy"] != "Not Collected"]
    clinical_data = clinical_data[clinical_data["Radiation"] != "Not Collected"]
    clinical_data = clinical_data[clinical_data["Recurrence"] != "Not Collected"]
    clinical_data["Pack Years"] = clinical_data["Pack Years"].fillna(0)
    clinical_data = clinical_data.set_index("Case ID")

    clinical_dummy = pd.get_dummies(clinical_data[["Age at Histological Diagnosis","Weight (lbs)","Gender","Smoking status","Pack Years","Chemotherapy","Radiation","Recurrence","Survival Status","Days Survived", "Pathological T stage", "Pathological N stage", "Pathological M stage"]], columns=["Smoking status", "Pathological T stage", "Pathological N stage", "Pathological M stage"])

    for x in ["Gender", "Survival Status","Chemotherapy","Radiation","Recurrence"]:
      factorized = pd.factorize(clinical_dummy[x])
      clinical_dummy[x] = factorized[0]
      print(x, factorized[1])

    return clinical_dummy
    
  def clinical_train(self, clinical_sample):

    row = clinical_sample.copy()

    row["Age at Histological Diagnosis"] = int(float(row["Age at Histological Diagnosis"]) + float(row["Age at Histological Diagnosis"])*np.random.uniform(-0.1,0.1))
    row["Weight (lbs)"] = int(float(row["Weight (lbs)"]) + float(row["Weight (lbs)"])*np.random.uniform(-0.1,0.1))
    row["Pack Years"] = int(float(row["Pack Years"]) + float(row["Pack Years"])*np.random.uniform(-0.1,0.1))
    row["Days Survived"] = int(float(row["Days Survived"]) + float(row["Days Survived"])*np.random.uniform(-0.01,0.01))

    days_survived = torch.Tensor([row["Days Survived"]])
    survival_status = torch.Tensor([row["Survival Status"]])

    clinical_input = row.drop(["Survival Status", "Days Survived"]).astype(float).to_numpy()
    clinical_input = torch.from_numpy(clinical_input).to(torch.float32)

    return clinical_input, days_survived, survival_status
  
  def clinical_test(self, clinical_sample):

    row = clinical_sample.copy()

    days_survived = torch.Tensor([row["Days Survived"]])
    survival_status = torch.Tensor([row["Survival Status"]])

    clinical_input = row.drop(["Survival Status", "Days Survived"]).astype(float).to_numpy()
    clinical_input = torch.from_numpy(clinical_input).to(torch.float32)

    return clinical_input, days_survived, survival_status
  
  # def RNA_train(self,RNA_sample):
  #   row = RNA_sample.copy()
  #   row = row.apply(lambda x: x + x*np.random.uniform(-0.001,0.001)).astype(float).to_numpy()
  #   row = torch.from_numpy(row).to(torch.float32)
  #   return row
  
  def RNA_test(self,RNA_sample):
    row = RNA_sample.copy()
    row = row.astype(float).to_numpy()
    row = torch.from_numpy(row).to(torch.float32)
    return row

  def radiomic_train(self,radiomic_sample):
    row = radiomic_sample.copy()
    row = row.apply(lambda x: x + x*np.random.uniform(-0.001,0.001)).astype(float).to_numpy()/self.radiomic_train_means.astype(float).to_numpy()
    row = torch.from_numpy(row).to(torch.float32)
    return row

  def radiomic_test(self,radiomic_sample):
    row = radiomic_sample.copy()
    row = row.astype(float).to_numpy()/self.radiomic_train_means.astype(float).to_numpy()
    row = torch.from_numpy(row).to(torch.float32)
    return row

  def __len__(self):
    if self.phase == "train":
      return len(self.x_train)
    
    elif self.phase == "train_val":
      return len(self.x_train)
    
    elif self.phase == "val":
      return len(self.x_val)
    
    elif self.phase == "test":
      return len(self.x_test)
  
  def __getitem__(self,idx):

    output = []

    if self.phase == "train" or self.phase == "train_val" or self.phase == "train_frozen":

      if self.x_train[idx] in self.additional_ct:
        CT_raw = np.load(self.additional_folder+"CT/"+self.x_train[idx])
        seg_raw = np.load(self.additional_folder+"seg/"+self.y_train[idx])

      else:
        CT_raw = np.load(self.data_root+"CT/"+self.x_train[idx])
        seg_raw = np.load(self.data_root+"seg/"+self.y_train[idx])

      CT_sample = self.data_process(CT_raw,seg_raw)

      data_name = self.x_train[idx][:7]
      
      if self.phase == "train":
        clinical_input, days_survived, survival_status = self.clinical_train(self.clinical_data.loc[data_name,:])
      else:
        clinical_input, days_survived, survival_status = self.clinical_test(self.clinical_data.loc[data_name,:])

    elif self.phase == "val":
      CT_raw = np.load(self.data_root+"CT/"+self.x_val[idx])
      seg_raw = np.load(self.data_root+"seg/"+self.y_val[idx])

      CT_sample = self.data_process(CT_raw,seg_raw)

      data_name = self.x_val[idx][:7]

      clinical_input, days_survived, survival_status = self.clinical_test(self.clinical_data.loc[data_name,:])
    
    elif self.phase == "test":
      CT_raw = np.load(self.data_root+"CT/"+self.x_test[idx])
      seg_raw = np.load(self.data_root+"seg/"+self.y_test[idx])

      CT_sample = self.data_process(CT_raw,seg_raw)

      data_name = self.x_test[idx][:7]

      clinical_input, days_survived, survival_status = self.clinical_test(self.clinical_data.loc[data_name,:])
    
    if self.args.classifier == True:
      if survival_status.item() == 0:
        survival_status = torch.Tensor([1,0])
      elif survival_status.item() == 1:
        survival_status = torch.Tensor([0,1])
    
    output.extend([CT_sample, clinical_input])

    if self.args.rna == True:
      RNA_sample = self.RNA_test(self.RNA.loc[data_name,:])
      output.append(RNA_sample)
    
    if self.args.radiomic == True:
      if self.phase == "train":
        radiomic_sample = self.radiomic_train(self.radiomic_data.loc[data_name,:])
      else:
        radiomic_sample = self.radiomic_test(self.radiomic_data.loc[data_name,:])
      output.append(radiomic_sample)
    
    output.extend([days_survived, survival_status])
    return output

  def data_process(self, CT, seg):

    CT_output, seg_output = self.rotate_data(CT, seg)

    if self.phase == "train" or self.phase == "train_frozen":
      CT_output, seg_output = self.train_crop(CT_output, seg_output)
    else:
      CT_output, seg_output = self.test_crop(CT_output, seg_output)

    CT_output = self.resize_data(CT_output)
    seg_output = self.resize_data(seg_output)
  
    CT_output = self.normalize(CT_output)

    CT_output = self.reshape_data(CT_output)
    seg_output = self.reshape_data(seg_output)

    seg_output = self.seg_cleaning(seg_output)

    output = np.concatenate([CT_output, seg_output], axis=0)

    output = self.post_processing(output)

    return output

  def rotate_data(self, data, label):

    random_angle = np.random.uniform(-20,20)

    data_rotate = ndimage.rotate(data,
                               random_angle,
                               axes=(1,2),
                               reshape=False,
                               cval=data.min(),
                               order=3)
    
    label_rotate = ndimage.rotate(label,
                                  random_angle,
                                  axes=(1,2),
                                  reshape=False,
                                  cval=label.min(),
                                  order=3)
    
    return data_rotate, label_rotate

  def train_crop(self, data, label):
      """
      Random crop
      """
      target_indexs = np.where(label>=0.5)
      [img_d, img_h, img_w] = data.shape
      [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
      [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
      [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
      Z_min = int((min_D - target_depth*1.0/2) * np.random.uniform(0,1))
      Y_min = int((min_H - target_height*1.0/2) * np.random.uniform(0,0.5))
      X_min = int((min_W - target_width*1.0/2) * np.random.uniform(0,0.5))
      
      Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * np.random.uniform(0,1)))
      Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * np.random.uniform(0,0.5)))
      X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * np.random.uniform(0,0.5)))
      
      Z_min = np.max([0, Z_min])
      Y_min = np.max([0, Y_min])
      X_min = np.max([0, X_min])

      Z_max = np.min([img_d, Z_max])
      Y_max = np.min([img_h, Y_max])
      X_max = np.min([img_w, X_max])

      Z_min = int(Z_min)
      Y_min = int(Y_min)
      X_min = int(X_min)
      
      Z_max = int(Z_max)
      Y_max = int(Y_max)
      X_max = int(X_max)

      return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]


  def test_crop(self, data, label):
      """
      Random crop
      """
      [img_d, img_h, img_w] = data.shape

      Z_min = int(img_d/10)
      Y_min = int(img_h/10)
      X_min = int(img_w/10)
      
      Z_max = img_d - Z_min
      Y_max = img_h - Y_min
      X_max = img_w - X_min

      return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

  def resize_data(self,data):
    """
    Resize the data to the input size
    """ 
    [depth, height, width] = data.shape
    scale = [self.input_d/depth, self.input_h/height, self.input_w/width]  
    data = ndimage.zoom(data, scale, order=1)

    return data

  def normalize(self,data): 
    pixels = data[data > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (data - mean)/std
    return out

  def reshape_data(self,data):
    [z, y, x] = data.shape
    data_output = np.reshape(data, [1, z, y, x])
    return data_output
  
  def seg_cleaning(self, data):
    """
    create the negative of the label and appends it to the array in dimension 0
    """
    where_high = np.where(data >= 0.5)
    where_low = np.where(data < 0.5)

    data[where_high] = 1
    data[where_low] = 0

    return data
  
  def post_processing(self,data):
    data_output = data.astype("float32")
    data_output = torch.from_numpy(data_output)
    return data_output
