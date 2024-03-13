import os
import numpy as np
import pydicom as dicom
import pandas as pd
from matplotlib import pyplot as plt, animation

directories = "NSCLC Radiogenomics"

xyz_dicts = dict()

for dir in os.listdir(directories):

    CT1_dict = dict()
    seg_dict = dict()
    xyz_dict = dict()

    subdir = os.listdir(f"{directories}\\{dir}\\")[0]
    subdir_path = f"{directories}\\{dir}\\{subdir}"

    for sub_subdir in os.listdir(subdir_path):

        if "segmentation" in sub_subdir:

            seg_file = dicom.dcmread(f"{subdir_path}\\{sub_subdir}\\1-1.dcm",force=True)

            for i in range(seg_file.NumberOfFrames): # crea dizionario[coordinate x,y,z] = immagine corrispondente
                
                seg_position = seg_file[0x52009230][i].PlanePositionSequence[0].ImagePositionPatient
                seg_dict[tuple(seg_position)] = seg_file.pixel_array[i,:,:]
            
            xy = seg_file[0x52009229][0].PixelMeasuresSequence[0].PixelSpacing
            z = seg_file[0x52009229][0].PixelMeasuresSequence[0].SliceThickness
            xyz_dict["x"] = float(xy[0])
            xyz_dict["y"] = float(xy[1])
            xyz_dict["z"] = float(z)
            xyz_dicts[dir] = xyz_dict
        
        else:
            
            for file in os.listdir(f"{subdir_path}\\{sub_subdir}"):

                image = dicom.dcmread(f"{subdir_path}\\{sub_subdir}\\{file}",force=True) # apre l'immagine

                coord = tuple(image.ImagePositionPatient) #recupera coordinate dell'immagine

                CT1_dict[coord] = image.pixel_array # crea dizionario[coordinate x,y,z] = immagine corrispondente

                if coord not in seg_dict.keys(): # aggiungi immagine vuota all'altro dizionario se non c'è corrispondenza
                    seg_dict[coord] = np.zeros(list(seg_dict.values())[0].shape)

    coords = list(CT1_dict.keys())
    coords.sort()

    CT1_list = []
    seg_list = []

    for coord in coords:
        CT1_list.append(CT1_dict[coord])
        seg_list.append(seg_dict[coord])

    CT1_array = np.array(CT1_list)
    seg_array = np.array(seg_list)

    np.save(f"new_CT\\{dir}.npy",CT1_array)
    np.save(f"new_seg\\{dir}.npy",seg_array)

    print(dir,"done")
        

pd_xyz = pd.DataFrame(xyz_dicts).T

pd_xyz.to_csv("xyz.csv")

# fig, (ax1, ax2) = plt.subplots(1,2,constrained_layout=True)

# CT1 = np.load("new_CT/R01-059.npy")
# seg = np.load("new_seg/R01-059.npy")

# cnt = 0
# def update(frame):

#     global cnt,CT1,seg

#     ax1.clear()
#     ax2.clear()

#     fig.suptitle(cnt)

#     ax1.imshow(CT1[cnt,:,:])
#     ax1.set_title(cnt, fontdict={"fontsize":6})
#     ax2.imshow(seg[cnt,:,:])
#     ax2.set_title(cnt, fontdict={"fontsize":6})

#     if cnt < CT1.shape[0]-1:
#         cnt += 1
#     else:
#         print("done")

# ani = animation.FuncAnimation(
#     fig, update,
#     frames=CT1.shape[0],
#     interval = 100)

# # per salvare il video
# writervideo = animation.FFMpegWriter(fps=6)
# ani.save('boh.mp4', writer=writervideo)
# plt.close()







