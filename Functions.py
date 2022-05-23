import scaleapi
from scaleapi.tasks import TaskType
from scaleapi.api import Api
from scaleapi.tasks import TaskReviewStatus, TaskStatus

from skimage import io
from skimage import feature
from skimage.color import rgb2gray, rgba2rgb
from skimage import filters
from skimage.morphology import diamond, disk, closing, erosion, dilation, opening, rectangle

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def resize_Array(arr1, arr2):
    if arr1.shape[0] < arr2.shape[0]:
        arr1 = np.append(arr1, np.zeros((arr2.shape[0]-arr1.shape[0], ), dtype=int))
    elif arr1.shape[0] > arr2.shape[0]:
        arr1 = arr1[:-(arr1.shape[0]-arr2.shape[0]),:]

    if arr1.shape[1] < arr2.shape[1]:
        arr1 = np.c_[arr1, np.zeros((arr1.shape[0], arr2.shape[1]-arr1.shape[1]), dtype=int)]
    elif arr1.shape[1] > arr2.shape[1]:
        arr1 = arr1[:,:-(arr1.shape[1]-arr2.shape[1])]
    
    return arr1.astype(int)
    

def create_Rectangle(height, width):
    rect = rectangle(height, width)*0
    rect[0:2,:] = True
    rect[-2:,:] = True
    rect[:,:2] = True
    rect[:, -2:] = True
    
    return rect

def plot_images(original_image, filtered_image):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 25),
                         sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(original_image, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original Image', fontsize=20)

    ax[1].imshow(filtered_image, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Filtered Image', fontsize=20)

    fig.tight_layout()
    plt.show()

def get_API_tasks(key, project):
    client = scaleapi.ScaleClient(key)
    tasks = client.get_tasks(project_name = project)
    
    return list(tasks)

def audit_task(task_list, num_annotation=2, plot=False, dilation_=False, sigma=1, threshold=0.2):
    df = pd.DataFrame(columns=['uuid', 'label', 'rect_conf', 'dia_conf', 'dis_conf', 'warning', 'error'])
    
    task_list_copy = task_list.copy()
    task_to_check = random.choice(task_list_copy)
        
    image_numpy = io.imread(task_to_check.as_dict()['params']['attachment'])
    if image_numpy.shape[2] == 3:
        image_numpy = rgb2gray(image_numpy)
    else:
        image_numpy = rgb2gray(rgba2rgb(image_numpy))
        
    annotations = random.sample(task_to_check.as_dict()['response']['annotations'], num_annotation)
    
    for annotation in annotations:
        label = annotation['label']
        uuid = annotation['uuid']
        width = int(annotation['width'])
        height = int(annotation['height'])
        left = int(annotation['left'])
        top = int(annotation['top'])
        
        sub_image = image_numpy[top:top+height, left:left+width]
        
        if dilation_:
            edge = dilation(feature.canny(sub_image,sigma=sigma))
        else: 
            edge = feature.canny(sub_image,sigma=sigma)
        
        r = create_Rectangle(height, width)
        dia = feature.canny(diamond(height/2), use_quantiles=True)
        dis = feature.canny(disk(height/2), use_quantiles=True) 
        
        if plot:
            plt.imshow(image_numpy[top:top+height, left:left+width])
            plot_images(sub_image, edge)
        
        dia = resize_Array(dia, edge)
        dis = resize_Array(dis, edge)
        
        r_ = r + edge
        dia_ = dia + edge
        dis_ = dis + edge
        
        r_[r_<2] = 0
        dia_[dia_<2] = 0
        dis_[dis_<2] = 0
        
        r_[r_>=2] = 1
        dia_[dia_>=2] = 1
        dis_[dis_>=2] = 1
        
        if plot:
            plot_images(r, r_)
            plot_images(dia, dia_)
            plot_images(dis, dis_)
        
        rect_conf = r_.mean()/r.mean()
        dia_conf = dia_.mean()/dia.mean()
        dis_conf = dis_.mean()/dis.mean()
        
        rect_conf_b = rect_conf < threshold
        dia_conf_b = dia_conf < threshold
        dis_conf_b = dia_conf < threshold
        
        warning = rect_conf_b or dia_conf_b or dis_conf_b
        error = rect_conf_b and dia_conf_b and dis_conf_b
        
        data = pd.DataFrame({'uuid':uuid, 'label':label, 'rect_conf':rect_conf, 
                'dia_conf':dia_conf, 'dis_conf':dis_conf, 'warning':warning, 'error':error}, index=[0])
        df = pd.concat([df,data])
        
    df.to_csv(task_to_check.as_dict()['task_id']+'.csv')
    print("CSV file has been created")