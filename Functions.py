from distutils.log import warn
from email import message
import scaleapi
from scaleapi.tasks import TaskType
from scaleapi.api import Api
from scaleapi.tasks import TaskReviewStatus, TaskStatus

from skimage import io
from skimage import feature
from skimage.color import rgb2gray, rgba2rgb
from skimage import filters
from skimage.transform import rescale
from skimage.morphology import closing, erosion, dilation, opening, rectangle

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def resize_Array(arr1, arr2):
    '''
    Resize array to match shape of another array, 
    either by adding rows or columns of 0's,
    or removing them

            Parameters:
                    arr1 (np.ndarray): Array to reshape
                    arr2 (np.ndarray): Array whose shape we will match

            Returns:
                    arr1 (np.ndarray): Array that matches arr2's shape
    '''

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

    '''
    Create a matrix of the form (height, width), that represents a rectangle.
    With ts perimeter as 1's and everything else as 0's

            Parameters:
                    height (int): Height of the rectangle
                    width (int): Width of the rectangle

            Returns:
                    rect (np.ndarray): Array that contains the rectangle perimeter
    '''

    rect = rectangle(height, width)*0
    rect[0:2,:] = True
    rect[-2:,:] = True
    rect[:,:2] = True
    rect[:, -2:] = True
    
    return rect

def plot_images(original_image, filtered_image):

    '''
    Plot two images side by side

            Parameters:
                    original_image (np.ndarray): First image
                    filtered_image (np.ndarray): Second image

    '''

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 25),
                         sharex=False, sharey=False)

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

    '''
    Get all tasks from a project using ScaleAPI

            Parameters:
                    key (string): Users API key
                    project (string): Name of the project on Scale's platform

            Returns:
                    list(tasks) (list): List that contains the project's tasks
    '''    

    client = scaleapi.ScaleClient(key)
    tasks = client.get_tasks(project_name = project)
    
    return list(tasks)

def audit_task(task_list, plot=False, sigma=1, threshold=0.055):

    '''
    Grab a task at random and obtain an specific number of annotations to check
    Use a Canny filter to obtain the edges of each annotation and create certain figures (rectangle, diamond, disk),
    which is the shape of most street signs, of the size of the bounding box
    Sum these matrices and see if any of these shapes are contained in the part of the image delimited by its 
    annotation bounding box

            Parameters:
                    task_list (list): List containing the tasks of a project
                    plot (bool): Flag used to plot the images or not
                    sigma (int): Value used for the canny filter. Max value is 3
                    threshold (float): Limit to use to decide if we have warnings and errors

            Returns:
                    CSV containing the annotation uuid, label, the confidence for each shape, warning and error
    '''  

    df = pd.DataFrame(columns=['task_id', 'uuid', 'label', 'message'])
    
    task_list_copy = task_list.copy()
    # task_to_check = random.choice(task_list_copy)

    for task_to_check in task_list_copy:
        
        image_numpy = io.imread(task_to_check.as_dict()['params']['attachment'])
        if image_numpy.shape[2] == 3:
            image_numpy = rgb2gray(image_numpy)
        else:
            image_numpy = rgb2gray(rgba2rgb(image_numpy))
            
        annotations = task_to_check.as_dict()['response']['annotations']
        for annotation in annotations:
            label = annotation['label']
            uuid = annotation['uuid']
            width = int(annotation['width'])
            height = int(annotation['height'])
            left = int(annotation['left'])
            top = int(annotation['top'])
            
            sub_image = image_numpy[top:top+height, left:left+width]
            edge = feature.canny(rescale(sub_image, 2, 2, anti_aliasing=True, anti_aliasing_sigma=1.25),sigma=sigma)
            
            if plot:
                plot_images(rescale(sub_image, 2, 2, anti_aliasing=True, anti_aliasing_sigma=1.25), edge)
            
            num_of_pixels = edge.sum()
            mean_of_image = edge.mean()

            warning = mean_of_image < threshold
            error = not (max(height, width) <= num_of_pixels < height*width)
            
            if warning and error:
                message = "This annotation does not contain an object inside it"
            elif warning and not error:
                message = "Please check this annotation, it might be wrong"  
            else:
                message = "This annotation looks good"          
            
            data = pd.DataFrame({'task_id': task_to_check.as_dict()['task_id'], 'uuid':uuid, 'label':label, 
            'message':message}, index=[0])
            df = pd.concat([df,data])
            
    df.to_csv('Linter.csv')
    print("CSV file has been created")