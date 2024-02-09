from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from PyQt5 import QtCore
import numpy as np

from TimeFrame import TimeFrame
import globals


def get_coordinates_with_slider(image, cmap = "gray"):
    """ Get coordinates of corners US image and depth

    Args:
        image (ndarray): image out of list images
        cmap (str, optional): Colormap used for imshow. Defaults to "gray".

    Returns:
        coordinates (list): upper left and upper right coordinates of corners US image
        depth (float): depth of US image
    """
    while True:
        fig = plt.figure()
        gs = GridSpec(2, 2, figure = fig, height_ratios = [4,1])
        axs = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,:])]
        plot = axs[0].imshow(image, cmap = cmap)
        slider_depth = Slider(axs[1], "Depth of US image", valmin = 0, valmax = 7, valinit = 0, valstep = 0.5)
        fig.suptitle("Get upper left and upper right coordinate and fill in depth")
        plt.show(block=False)
        win =hide_close_button(fig)
        coordinates = plt.ginput(0, timeout = 0, mouse_add=MouseButton.LEFT, mouse_pop= MouseButton.RIGHT, mouse_stop=None)
        plt.close("all")
        
        return [tuple(int(x) for x in st) for st in coordinates], float(slider_depth.val)
    
def hide_close_button(fig):
    """ Disables the closing button of a window

    Args:
        fig (matlibplot.figure): the figure in which the close buttons has to be disabled

    Returns:
        matlibplot.figure: the new figure with the disabled closing button
    """
    win = fig.canvas.manager.window
    win.setWindowFlags(win.windowFlags() | QtCore.Qt.CustomizeWindowHint)
    win.setWindowFlags(win.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
    return win

def convert_str_to_tuple(value):
    """ Convert a string to a tuple, given it's in right format

    Args:
        value (str): string which has to be converted into a tuple

    Returns:
        tuple: the extracted tuple
    """
    value = value.replace("(", "").replace(")", ""). replace(" ", "")
    value_tuple = tuple(map(lambda x: int(float(x)), value.split(",")))
    return value_tuple

def corner_depth_popup(coord1, coord2, depth):
    """ Creates a popup where user has to fill in the corner coordinates and the depth

    Args:
        coord1 (str): The coordinate left upper corner in tuple format
        coord2 (str): The coordinate right upper corner in tuple format
        depth (str): The depth setting of the US scanner

    Returns:
        tuple, tuple, float: the 3 values converted to right type after user has filled in right details
    """
    fig, axs = plt.subplots(3, 1)
    fig.suptitle("Fill in skipped variables")
    coord1_box = TextBox(axs[0], label = "Upper left corner (x1,y1)?", initial = coord1)
    coord2_box = TextBox(axs[1], label = "Upper right corner (x1,y1)?", initial = coord2)
    depth_box = TextBox(axs[2], label = "Please fill in the depth of the ultrasound image in cm (D): ", initial = depth)
    plt.tight_layout()
    plt.show(block=False)
    win = hide_close_button(fig)
    plt.ginput(0, timeout = 0, mouse_add= MouseButton.RIGHT, mouse_pop= MouseButton.MIDDLE, mouse_stop= MouseButton.BACK)
    plt.close("all")
    return convert_str_to_tuple(coord1_box.text), convert_str_to_tuple(coord2_box.text), float(depth_box.text)

def update_all(images, images_masked, sliders,plots,setcor, diff_between_xlines=20):

    
    """ Onchange sliders changes the 'super slider', updates everything on each change, except in if statement

    update_all(axs, images, [time_change_slider], [org_image, globals.line_crop_x, masked_image])

    Args:
        axs (list): all axis in 'super slider'
        images (ndarray): the denoised images
        sliders (list): all sliders in 'super slider'
        plots (list): all plots in 'super slider'
        contrast_or_timeframe_changed (bool, optional): if the contrast or timeframe has changed, speeds up program, because not all the time heavy calculations. Defaults to False.
        angle_changed (bool, optional):  if the m mode angle has changed, speeds up program, because not all the time heavy calculations. Defaults to False.
    """
    

    
    time_change_slider = sliders[0]
    org_image, masked_image = plots

    time = int(time_change_slider.val)

    slider_linex=sliders[1]
    slider_liney1=sliders[2]
    slider_liney2=sliders[3]

    setcory=setcor[1]
    setcorx=setcor[0]


    new_x_coordinate1=int(slider_linex.val)
    new_y_coordinate1=int(slider_liney1.val)
    new_y_coordinate2=int(slider_liney2.val)

    #diff_between_xlines=20

    new_x_coordinate2=diff_between_xlines+new_x_coordinate1
    

    # Update original image
    org_image.set_data(images[time])
    
    
    
    #update line
    globals.line_crop_x.set_data([[new_x_coordinate1],[new_x_coordinate1]], setcory)
    globals.line_crop_x2.set_data([[new_x_coordinate2],[new_x_coordinate2]], setcory)
    globals.line_crop_y.set_data(setcorx,[[new_y_coordinate1],[new_y_coordinate1]])
    globals.line_crop_y2.set_data(setcorx,[[new_y_coordinate2],[new_y_coordinate2]])

    # Update cropped masked image
    masked_image.set_data(images_masked[time,new_y_coordinate1:new_y_coordinate2,new_x_coordinate1:new_x_coordinate2])

    #update crop
    globals.cropped_image.set_data(images[time,new_y_coordinate1:new_y_coordinate2,new_x_coordinate1:new_x_coordinate2])

   

    return diff_between_xlines
def delete_current_image():


    rows=globals.images_masked[globals.current_time].shape[0]
    columns=globals.images_masked[globals.current_time].shape[1]
    nan_slice=np.zeros(shape=(1,rows,columns))
    globals.images_masked[globals.current_time,:,:]=nan_slice
    print('image deleted')

def divide_for_limits():


    next_time_frame = TimeFrame(globals.current_time, globals.current_time_frame.end_frame, globals.current_time_frame.lower_bound, globals.current_time_frame.upper_bound)
    globals.current_time_frame.end_frame = globals.current_time
    globals.time_frames.insert(globals.time_frames.index(globals.current_time_frame) + 1, next_time_frame)
    print('New time frame created')
    globals.current_time_frame = next_time_frame
    
# def on_key_press(key):

#     try:
#         key_char = key.char
#         if key_char == 'd':
#             delete_current_image()
            
#     except AttributeError:
#         pass

def on_c_press(key):

    key_char = key.char
    if key_char == 'c':
        divide_for_limits()
    if key_char == 'd':
        delete_current_image()    


def find_pixelvalue(image, depth_cm):
    depth_in_mm = depth_cm * 10
    pixel_amount_y = image.shape[0]
    return depth_in_mm / pixel_amount_y

def calculating_distance(images,depth,images_masked,mid_col,upperlimit,lowerlimit):

    distance_in_pixels=[]
    for image in images_masked:
            upper_nonzeros=[]
            lower_nonzeros=[]
            for n in np.arange(upperlimit, 0, -1):
                if image[n,mid_col]!=0:
                    upper_nonzeros.append(n)
            upper_nonzero_array=np.array(upper_nonzeros)
            for n in np.arange(lowerlimit, len(images_masked[0, :, 0]), 1):
                if image[n,mid_col]!=0:
                    lower_nonzeros.append(n)
            lower_nonzero_array=np.array(lower_nonzeros)
            if len(upper_nonzero_array)>0 and len(lower_nonzero_array>0):
                pixel_distance = lower_nonzero_array[0] - upper_nonzero_array[0]
                distance_in_pixels.append(pixel_distance)
            else:   
                distance_in_pixels.append(np.nan)
    pixel_size = find_pixelvalue(images[0], depth)
    distance_in_mm = [pixel_size * distance for distance in distance_in_pixels]
    return distance_in_mm

def update_for_thickness(images,depth,type, cropped_images,images_masked,sliders: list[Slider], setcor,mid_col,length_frame):


    if globals.ignore_slider_changed:
        return
    
    end_interval=len(images_masked)
    
    # defining sliders
    time_change_slider_thick=sliders[0]
    slider_line_upper_limit=sliders[1]
    slider_line_lower_limit=sliders[2]
    
    #values
    globals.current_time = int(time_change_slider_thick.val)
    upper_limit=int(slider_line_upper_limit.val)
    lower_limit=int(slider_line_lower_limit.val)

    if type == 'time':
        for time_frame in globals.time_frames:
            if(globals.current_time >= time_frame.begin_frame and globals.current_time < time_frame.end_frame):
                globals.current_time_frame = time_frame
                break
        globals.ignore_slider_changed = True
        slider_line_upper_limit.set_val(globals.current_time_frame.upper_bound)
        slider_line_lower_limit.set_val(globals.current_time_frame.lower_bound)
        globals.ignore_slider_changed = False
        upper_limit = globals.current_time_frame.upper_bound
        lower_limit = globals.current_time_frame.lower_bound
    else:
        globals.current_time_frame.upper_bound = upper_limit
        globals.current_time_frame.lower_bound = lower_limit      


    #update cropped image
    globals.masked_image_thick.set_data(images_masked[globals.current_time])
    globals.cropped_image_thick.set_data(cropped_images[globals.current_time])

    #update lines
    globals.line_upper_limit.set_data(setcor,[upper_limit,upper_limit])
    globals.line_lower_limit.set_data(setcor,[lower_limit, lower_limit])
    globals.line_time_thick.set_data([globals.current_time*length_frame,globals.current_time*length_frame],[0, 5])

    globals.distance_in_mm_cal=[]
    for time_frame in globals.time_frames:
        globals.distance_in_mm_cal_intermediate = calculating_distance(images,depth,images_masked[range(time_frame.begin_frame,time_frame.end_frame,1)],mid_col,time_frame.upper_bound,time_frame.lower_bound)
        globals.distance_in_mm_cal = np.concatenate((globals.distance_in_mm_cal, np.array(globals.distance_in_mm_cal_intermediate)))
    

    globals.time_axis_cal = np.arange(0, len(globals.distance_in_mm_cal), 1)*length_frame

    globals.distanceplot.set_data(globals.time_axis_cal, globals.distance_in_mm_cal)
    values_without_nan = [value for value in globals.distance_in_mm_cal if not np.isnan(value)]

    globals.axs_for_lim.set_xlim(min(globals.time_axis_cal), max(globals.time_axis_cal))


    if len(values_without_nan)>0:
        globals.axs_for_lim.set_ylim((min(values_without_nan)-0.2), (max(globals.distance_in_mm_cal)+0.2))
    else:
        globals.axs_for_lim.set_ylim(0, 3)
   


    
    return