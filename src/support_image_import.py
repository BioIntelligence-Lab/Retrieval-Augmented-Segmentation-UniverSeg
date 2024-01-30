from torchvision import transforms
from PIL import Image
import torch
import os
from medsimilarity2 import hash_comparison,structural_comparison1,dense_vector_comparison,lpips_comparison
import datetime
import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A

# Function to get a list of files in a folder
def get_files_in_folder(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except Exception as e:
        return str(e)
    
# Function to import support images and perform hash comparison
def import_support_image_hash(num,img_path,mask_path,final_image_path,device,size = (200,200)):
    """
    Imports support images, performs hash comparison, and returns processed tensors.

    Parameters:
    - num (int): Number of support images to consider.
    - img_path (str): Path to the support images folder.
    - mask_path (str): Path to the support masks folder.
    - final_image_path (str): Path to the final image.
    - device: Torch device for processing.
    - size (tuple): Desired size of the images.

    Returns:
    - tuple: Tensors containing processed support images and masks.
    """
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size)])

    test_num = num + 5

    transform2 = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

    img_ran = Image.open(final_image_path)

    folder_path1 = img_path

    file_list1 = get_files_in_folder(img_path)

    for i in range(len(file_list1)):
        file_list1[i] = folder_path1 + '/' + file_list1[i]
    
    file_list = hash_comparison(final_image_path, file_list1, top_k=test_num)
    img_ran = transform(Image.open(final_image_path))
    
    file_list1 = file_list[1:,0]
    exp1 = []
    for i in range(num):
        img = Image.open(folder_path1 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp1.append(img)
    arr1 = torch.stack(exp1, axis=0)
    folder_path2 = mask_path
    file_list2 = get_files_in_folder(folder_path2)
    exp2 = []
    for i in range(num):
        img = Image.open(folder_path2 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp2.append(img)
    arr2 = torch.stack(exp2, axis=0)

    img_ran = transform2(img_ran[:3, :, :])
    return arr1,arr2,img_ran

# Function to import support images and perform structural comparison

def import_support_image_structural(num,img_path,mask_path,final_image_path,device,size = (200,200)):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size)])

    test_num = num+5

    transform2 = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

    img_ran = Image.open(final_image_path)

    folder_path1 = img_path

    file_list1 = get_files_in_folder(img_path)

    for i in range(len(file_list1)):
        file_list1[i] = folder_path1 + '/' + file_list1[i]
    
    file_list = structural_comparison1(final_image_path, file_list1, top_k=test_num)
    
    img_ran = transform(Image.open(final_image_path))
    
    file_list1 = file_list[1:,0]
    exp1 = []
    for i in range(num):
        img = Image.open(folder_path1 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp1.append(img)
    arr1 = torch.stack(exp1, axis=0)
    folder_path2 = mask_path
    file_list2 = get_files_in_folder(folder_path2)
    exp2 = []
    for i in range(num):
        img = Image.open(folder_path2 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp2.append(img)
    arr2 = torch.stack(exp2, axis=0)

    img_ran = transform2(img_ran[:3, :, :])
    return arr1,arr2,img_ran
# Function to import support images and perform dense vector comparison

def import_support_image_dense_vector(num,img_path,mask_path,final_image_path,device,size = (200,200)):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size)])

    test_num = num+5

    transform2 = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

    img_ran = Image.open(final_image_path)

    folder_path1 = img_path

    file_list1 = get_files_in_folder(img_path)

    for i in range(len(file_list1)):
        file_list1[i] = folder_path1 + '/' + file_list1[i]
    file_list = dense_vector_comparison(final_image_path, file_list1, top_k=test_num)
    
    img_ran = transform(Image.open(final_image_path))
    
    file_list1 = file_list[1:,0]
    exp1 = []
    for i in range(num):
        img = Image.open(folder_path1 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp1.append(img)
    arr1 = torch.stack(exp1, axis=0)
    folder_path2 = mask_path
    file_list2 = get_files_in_folder(folder_path2)
    exp2 = []
    for i in range(num):
        img = Image.open(folder_path2 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp2.append(img)
    arr2 = torch.stack(exp2, axis=0)

    img_ran = transform2(img_ran[:3, :, :])
    return arr1.to('cpu'),arr2.to('cpu'),img_ran.to('cpu'),

# Function to import support images, perform hash comparison with augmentation

def import_support_image_hash_aug(num,img_path,mask_path,final_image_path,device,size = (200,200)):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size)])

    test_num = num+5

    transform2 = transforms.Grayscale(num_output_channels=1)


    aug = A.Compose([
        
    A.RandomSizedCrop(min_max_height=(size[0] - 15, size[0] - 5),height=size[0],width=size[1], p=0.5),
    A.CLAHE(p=0.8),
    A.RandomBrightnessContrast(p=0.8),    
    A.RandomGamma(p=0.8)])
    folder_path1 = img_path

    file_list1 = get_files_in_folder(img_path)
    for i in range(len(file_list1)):
        file_list1[i] = folder_path1 + '/' + file_list1[i]
    file_list = dense_vector_comparison(final_image_path, file_list1, top_k=5)
    img_ran = transform(Image.open(final_image_path))
    img_ran = transform2(img_ran)
    file_list1 = file_list[1:,0]
    folder_path2 = mask_path
    exp1 = []
    exp2 = []
    image = cv2.imread(folder_path1 +"/"+ file_list1[0],cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,size)
    masks = cv2.imread(folder_path2 +"/"+ file_list1[0],cv2.IMREAD_GRAYSCALE)
    masks = cv2.resize(masks,size)
    for i in range(num):
        augmented = aug(image=image, mask=masks)
        image_medium = augmented['image']
        mask_medium = augmented['mask']
        img = transform(image_medium)
        img = transform2(img[:3,:,:])
        exp1.append(img)
        mask = transform(mask_medium)
        mask = transform2(mask[:3,:,:])
        exp2.append(mask)
    arr2 = torch.stack(exp2, axis=0)
    arr1 = torch.stack(exp1, axis=0)
    return arr1,arr2,img_ran

# Function to import support images and perform LPIPS comparison

def import_support_image_lpips(num,img_path,mask_path,final_image_path,device,size = (200,200)):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(size)])

    test_num = num+5

    transform2 = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

    img_ran = Image.open(final_image_path)

    folder_path1 = img_path

    file_list1 = get_files_in_folder(img_path)

    for i in range(len(file_list1)):
        file_list1[i] = folder_path1 + '/' + file_list1[i]
    file_list = lpips_comparison(final_image_path, file_list1, top_k=test_num)
    img_ran = transform(Image.open(final_image_path))
    
    file_list1 = file_list[1:,0]
    exp1 = []
    for i in range(num):
        img = Image.open(folder_path1 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp1.append(img)
    arr1 = torch.stack(exp1, axis=0)
    folder_path2 = mask_path
    file_list2 = get_files_in_folder(folder_path2)
    exp2 = []
    for i in range(num):
        img = Image.open(folder_path2 +"/"+ file_list1[i])
        img = transform(img)
        img = transform2(img[:3,:,:])
        exp2.append(img)
    arr2 = torch.stack(exp2, axis=0)

    img_ran = transform2(img_ran[:3, :, :])
    return arr1,arr2,img_ran