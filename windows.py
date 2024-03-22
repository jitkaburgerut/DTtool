import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pydicom
from TimeFrame import TimeFrame
from helper import get_coordinates_with_slider, corner_depth_popup, update_all, hide_close_button, update_for_thickness
from multiprocessing.dummy import Pool
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, TextBox
from matplotlib.backend_bases import MouseButton
import numpy as np
import statistics
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.marker import Marker
import openpyxl as op
import globals 

def get_echo_images(filename = "", coordinates = [], depth = 0):
   
    # Check if file name is predefined, if not, ask for it - https://stackoverflow.com/questions/3579568/choosing-a-file-in-python-with-simple-dialog
    if not len(filename):  
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    
    # 2 types of files are correct, .DCM and .npz files, end up with globals.images
    if ".DCM" in filename:
        # import dicom and extract pixel array
        file = pydicom.dcmread(filename)


        globals.images = file.pixel_array
        cine_rate = int(file.CineRate)

        # Convert YBR to RGB when 
        if file.PhotometricInterpretation == "YBR_FULL_422":
            globals.images = pydicom.pixel_data_handlers.util.convert_color_space(globals.images, 'YBR_FULL_422', 'RGB')
        else:
            raise ValueError(f"Imported file has a unknown color schema: {file.PhotometricInterpretation}")

        # Color code globals.images from rgb to gray
        globals.images = np.dot(globals.images, [0.2989, 0.5870, 0.1140])

        # Get coördinates and depth from ultrasound of first image
        if len(coordinates) == 2:
            coords = coordinates
        else:
            coords, depth = get_coordinates_with_slider(globals.images[0])
        
        if len(coords) < 2 or depth <= 0:
            if len(coords) < 1:
                coords = [(0,0), (0,0)]
            elif len(coords) < 2:
                coords.append((0,0))
            elif len(coords) > 2:
                coords = coords[:2]
            coord1, coord2, depth = corner_depth_popup(str(coords[0]), str(coords[1]), str(depth))
        else:
            coord1, coord2 = coords[0], coords[1]
        
        # Check order of coord1 and coord2
        if coord1[0] < coord2[0]:
            x_min = coord1[0]
            x_max = coord2[0]
        else:
            x_min = coord2[0]
            x_max = coord1[0]

        # Calculate average y coord
        y_min = int((coord1[1] + coord2[1]) / 2)
        

        # Resize globals.images with calculated parameters
        globals.images = globals.images[:,y_min:,x_min:x_max] # time, y, x

        pool = Pool(30)
        denoise_futures = []
        for image in globals.images:
            denoise_futures.append(pool.apply_async(cv2.fastNlMeansDenoising, [np.array(image).astype(np.uint8), None, 30, 7, 21]))
        globals.images = [x.get() for x in denoise_futures]
        globals.images = np.array(globals.images)

        return filename, globals.images,cine_rate, depth
    
def crop_and_canny(images):
    masked_images = []

    for image in images:
        # DESISION, clip array
        img_blur = cv2.GaussianBlur(image, (3,3), 0)
        edges = cv2.Canny(image=img_blur, threshold1=1, threshold2=200)
        masked_images.append(edges) 
    globals.images_masked = np.array(masked_images)

    fig = plt.figure()
    gs = GridSpec(5, 3, figure=fig, height_ratios= [15,1,1,1,1])
    axs = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]),fig.add_subplot(gs[3,:]),fig.add_subplot(gs[4,:])]
    axs[0].title.set_text("Original Image")
    axs[1].title.set_text("Cropped image")
    axs[2].title.set_text("Masked Crop")

    globals.setxcor=[0, images[0].shape[1] - 1]
    setycor=[0, images[0].shape[0] - 1]

    globals.mid_col=round(len(globals.images_masked[0,0,:])/2)
    mid_row=round(len(globals.images_masked[0,:,0])/2)

    globals.org_image = axs[0].imshow(images[0], cmap="gray")
    (globals.line_crop_x,) = axs[0].plot([100, 100], setycor, color = "red")
    (globals.line_crop_y,) = axs[0].plot( globals.setxcor,[100, 100], color = "red")
    (globals.line_crop_y2,) = axs[0].plot([110, 110], setycor, color = "red")
    (globals.line_crop_x2,) = axs[0].plot(globals.setxcor,[350, 350], color = "red")
    (cross_x,) = axs[2].plot([globals.mid_col-5, globals.mid_col+5], [mid_row, mid_row], color = "red")
    (crossy,) = axs[2].plot([globals.mid_col, globals.mid_col], [mid_row-5,mid_row+5], color = "red")

    globals.cropped_image=axs[1].imshow(images[0,0:int(setycor[1]),0:int(globals.setxcor[1])], cmap="gray")
    masked_image = axs[2].imshow(globals.images_masked[0], cmap="gray")

    # Create (range)slider
    time_change_slider = Slider(axs[3], "Time", valmin = 0, valmax = len(images) - 1)
    slider_linex= Slider(axs[4], "Cropping X", valmin = 0, valmax = images[0].shape[1] - 1)
    slider_liney= Slider(axs[5], "Cropping Y1", valmin = 0, valmax = images[1].shape[1] - 1)
    slider_liney2= Slider(axs[6], "Cropping Y2", valmin = 0, valmax = images[1].shape[1] - 1)

    # Create events (range)sliders
    time_change_slider.on_changed(lambda x: update_all(images, globals.images_masked, [time_change_slider,slider_linex, slider_liney,slider_liney2], [globals.org_image, masked_image],[globals.setxcor, setycor]))
    slider_linex.on_changed(lambda x: update_all(images, globals.images_masked, [time_change_slider,slider_linex, slider_liney,slider_liney2], [globals.org_image, masked_image],[globals.setxcor, setycor]))
    slider_liney.on_changed(lambda x: update_all(images, globals.images_masked, [time_change_slider,slider_linex, slider_liney,slider_liney2], [globals.org_image, masked_image],[globals.setxcor, setycor]))
    slider_liney2.on_changed(lambda x: update_all(images, globals.images_masked, [time_change_slider,slider_linex, slider_liney,slider_liney2], [globals.org_image, masked_image],[globals.setxcor, setycor]))

    
    plt.show(block=False)
    win = hide_close_button(fig) 
    plt.ginput(0, timeout = 0, mouse_add= MouseButton.RIGHT, mouse_pop= MouseButton.MIDDLE, mouse_stop= MouseButton.BACK)
    plt.close("all")

    return  globals.images_masked, int(slider_linex.val), int(slider_liney.val), int(slider_liney2.val)

def get_distance(images,images_masked,x,y1,y2, length_frame,depth):

    globals.ignore_slider_changed = False
    globals.time_frames = [TimeFrame(0,len(images_masked)-1, 0, 0 )]
    globals.current_time_frame = globals.time_frames[0]

    globals.low_lim_dist=1
    globals.up_lim_dist=2.5

    images_masked=images_masked[:,y1:y2,x:(x+20)]
    globals.cropped_images=images[:,y1:y2,x:(x+20)]
    globals.mid_col=round(len(images_masked[0,0,:])/2)
    mid_row=round(len(images_masked[0,:,0])/2)

    fig = plt.figure()
    gs = GridSpec(4, 3, figure=fig, height_ratios= [15,1,1,1])
    axs = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]), fig.add_subplot(gs[1,:]), fig.add_subplot(gs[2,:]),fig.add_subplot(gs[3,:])]
    
    globals.setx=[0, images_masked[0].shape[1] - 1]
    
    maxy=images_masked[1].shape[0] - 1
    axs[0].title.set_text("Cropped image")
    axs[1].title.set_text("Masked crop")
    axs[2].title.set_text("Diaphragm thickness over time")
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Distance (mm)')
    globals.masked_image_thick = axs[1].imshow(images_masked[0], cmap="gray")
    globals.cropped_image_thick = axs[0].imshow(globals.cropped_images[0], cmap="gray")

    globals.axs_for_lim=axs[2]

    time_axis= np.arange(0, len(images_masked), 1)
    distance_in_mm=np.ones(len(images_masked))*1.8
    (globals.line_upper_limit,) = axs[1].plot(globals.setx,[0,0], color = "green")
    (globals.line_lower_limit,) = axs[1].plot(globals.setx,[maxy,maxy], color = "green")
    (globals.line_time_thick,) = axs[2].plot([0,0],[0, 5], color = "red")

    (globals.distanceplot,)=axs[2].plot(time_axis,distance_in_mm, color='black')

    globals.setxlim=globals.axs_for_lim.set_xlim(0, time_axis[-1])
    globals.setylim=globals.axs_for_lim.set_ylim(globals.low_lim_dist,globals.up_lim_dist)
    

    #sliders
    globals.time_change_slider_thick = Slider(axs[3], "Time", valmin = 0, valmax = len(images_masked) - 1)
    globals.slider_line_upper_limit= Slider(axs[4], "Upperbound", valmin = 0, valmax = maxy)
    globals.slider_line_lower_limit= Slider(axs[5], "Lower bound", valmin = 0, valmax = maxy)

    #changes ons sliders
    globals.time_change_slider_thick.on_changed(lambda x: update_for_thickness(images,depth,'time',globals.cropped_images,images_masked,[globals.time_change_slider_thick,globals.slider_line_upper_limit,globals.slider_line_lower_limit], globals.setx, globals.mid_col, length_frame))
    globals.slider_line_upper_limit.on_changed(lambda x: update_for_thickness(images,depth,'upper',globals.cropped_images,images_masked,[globals.time_change_slider_thick,globals.slider_line_upper_limit,globals.slider_line_lower_limit], globals.setx, globals.mid_col,length_frame))
    globals.slider_line_lower_limit.on_changed(lambda x: update_for_thickness(images,depth,'lower',globals.cropped_images,images_masked,[globals.time_change_slider_thick,globals.slider_line_upper_limit,globals.slider_line_lower_limit], globals.setx, globals.mid_col,length_frame))
    
    win = hide_close_button(fig)
    plt.ginput(0, timeout = 0, mouse_add= MouseButton.RIGHT, mouse_pop= MouseButton.MIDDLE, mouse_stop= MouseButton.BACK)
    plt.close("all")
    

    plt.show() 



    return time_axis, distance_in_mm

def get_TF():

    time_distance=globals.distanceplot.get_xdata()
    distances=globals.distanceplot.get_ydata()

    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig, height_ratios= [15])
    axs = [fig.add_subplot(gs[0,0])]
    axs[0].plot(time_distance,distances, color='black')
    axs[0].set_title('Diaphragm distance over time')
    axs[0].set_ylabel('Thickness (mm)')
    axs[0].set_xlabel('Time (s)')

    total_clicks=2

    coordinates1 = plt.ginput(total_clicks, timeout = 0,show_clicks=True, mouse_add= MouseButton.LEFT, mouse_pop= MouseButton.MIDDLE, mouse_stop= MouseButton.RIGHT)
    coordinates2 = plt.ginput(total_clicks, timeout = 0, show_clicks=True,mouse_add= MouseButton.LEFT, mouse_pop= MouseButton.MIDDLE, mouse_stop= MouseButton.RIGHT)
    coordinates3 = plt.ginput(total_clicks, timeout = 0,show_clicks=True, mouse_add= MouseButton.LEFT, mouse_pop= MouseButton.MIDDLE, mouse_stop= MouseButton.RIGHT)

    coordinates1=np.array(coordinates1)
    coordinates2=np.array(coordinates2)
    coordinates3=np.array(coordinates3)



    peaks=[]
    valleys=[]

    if len(coordinates1)==2:
        if coordinates1[0,1]>coordinates1[1,1]:
            x_coordinate_click_peak=coordinates1[0,0]
            index_click_peak=np.abs(time_distance - x_coordinate_click_peak).argmin()
            
            if index_click_peak>9:
                peak_in_range=max(distances[(index_click_peak-10):(index_click_peak+10)])
                range=distances[(index_click_peak-10):(index_click_peak+10)]
            else: 
                peak_in_range=max(distances[0:(index_click_peak+10)])
                range=distances[0:(index_click_peak+10)]
            peaks.append(peak_in_range)
            indexp1= np.array(range).argmax()
            indexp1=index_click_peak+indexp1-10
            
            x_coordinate_click_valley=coordinates1[1,0]
            index_click_valley=np.abs(time_distance - x_coordinate_click_valley).argmin()
            range=distances[(index_click_valley-10):(index_click_valley+10)]
            valley_in_range=min(distances[(index_click_valley-10):(index_click_valley+10)])
            valleys.append(valley_in_range)
            indexv1=np.array(range).argmin()
            indexv1=index_click_valley+indexv1-10

        else:
            x_coordinate_click_peak=coordinates1[1,0]
            index_click_peak=np.abs(time_distance - x_coordinate_click_peak).argmin()
            range=distances[(index_click_peak-10):(index_click_peak+10)]
            peak_in_range=max(distances[(index_click_peak-10):(index_click_peak+10)])
            peaks.append(peak_in_range)
            indexp1=np.array(range).argmax()
            indexp1=index_click_peak+indexp1-10
            
            x_coordinate_click_valley=coordinates1[0,0]
            index_click_valley=np.abs(time_distance - x_coordinate_click_valley).argmin()
            range=distances[(index_click_valley-10):(index_click_valley+10)]
            valley_in_range=min(distances[(index_click_valley-10):(index_click_valley+10)])
            valleys.append(valley_in_range)
            indexv1=np.array(range).argmin()
            indexv1=index_click_valley+indexv1-10
        

    if len(coordinates2)==2:
        if coordinates2[0,1]>coordinates2[1,1]:
            x_coordinate_click_peak=coordinates2[0,0]
            index_click_peak=np.abs(time_distance - x_coordinate_click_peak).argmin()
            range=distances[(index_click_peak-10):(index_click_peak+10)]
            peak_in_range=max(distances[(index_click_peak-10):(index_click_peak+10)])
            peaks.append(peak_in_range)
            indexp2=np.array(range).argmax()
            indexp2=indexp2+index_click_peak-10
            
            x_coordinate_click_valley=coordinates2[1,0]
            index_click_valley=np.abs(time_distance - x_coordinate_click_valley).argmin()
            range=distances[(index_click_valley-10):(index_click_valley+10)]
            valley_in_range=min(distances[(index_click_valley-10):(index_click_valley+10)])
            valleys.append(valley_in_range)
            indexv2=np.array(range).argmin()
            indexv2=indexv2+index_click_valley-10

        else:
            x_coordinate_click_peak=coordinates2[1,0]
            index_click_peak=np.abs(time_distance - x_coordinate_click_peak).argmin()
            range=distances[(index_click_peak-10):(index_click_peak+10)]
            peak_in_range=max(distances[(index_click_peak-10):(index_click_peak+10)])
            peaks.append(peak_in_range)
            indexp2=np.array(range).argmax()
            indexp2=indexp2+index_click_peak-10
            
            x_coordinate_click_valley=coordinates2[0,0]
            index_click_valley=np.abs(time_distance - x_coordinate_click_valley).argmin()
            range=distances[(index_click_valley-10):(index_click_valley+10)]
            valley_in_range=min(distances[(index_click_valley-10):(index_click_valley+10)])
            valleys.append(valley_in_range)
            indexv2=np.array(range).argmin()
            indexv2=indexv2+index_click_valley-10


    
    if len(coordinates3)==2:
        if coordinates3[0,1]>coordinates3[1,1]:
            x_coordinate_click_peak=coordinates3[0,0]
            index_click_peak=np.abs(time_distance - x_coordinate_click_peak).argmin()
            range=distances[(index_click_peak-10):(index_click_peak+10)]
            peak_in_range=max(distances[(index_click_peak-10):(index_click_peak+10)])
            peaks.append(peak_in_range)
            indexp3=np.array(range).argmax()
            indexp3=indexp3+index_click_peak-10
            
            x_coordinate_click_valley=coordinates3[1,0]
            index_click_valley=np.abs(time_distance - x_coordinate_click_valley).argmin()
            range=distances[(index_click_valley-10):(index_click_valley+10)]
            valley_in_range=min(distances[(index_click_valley-10):(index_click_valley+10)])
            valleys.append(valley_in_range)
            indexv3=np.array(range).argmin()
            indexv3=indexv3+index_click_valley-10

        else:
            x_coordinate_click_peak=coordinates3[1,0]
            index_click_peak=np.abs(time_distance - x_coordinate_click_peak).argmin()
            range=distances[(index_click_peak-10):(index_click_peak+10)]
            peak_in_range=max(distances[(index_click_peak-10):(index_click_peak+10)])
            peaks.append(peak_in_range)
            indexp3=np.array(range).argmax()
            indexp3=indexp3+index_click_peak-10
            
            x_coordinate_click_valley=coordinates3[0,0]
            index_click_valley=np.abs(time_distance - x_coordinate_click_valley).argmin()
            range=distances[(index_click_valley-10):(index_click_valley+10)]
            valley_in_range=min(distances[(index_click_valley-10):(index_click_valley+10)])
            valleys.append(valley_in_range)
            indexv3=np.array(range).argmin()
            indexv3=indexv3+index_click_valley-10
       
    TF=[]



    peaks=np.array(peaks)
    valleys=np.array(valleys)

   
    if len(peaks)>0 and len(valleys)>0:
        TF1=(peaks[0]-valleys[0])/valleys[0]
        TF.append(TF1)
        meanTF=statistics.mean(TF)

    if len(peaks)>1 and len(valleys)>1:
        TF2=(peaks[1]-valleys[1])/valleys[1]
        TF.append(TF2)
        deviation_TF_1_2=2*(TF[0]-TF[1])/(TF[0]+TF[1])
        if abs(deviation_TF_1_2)>0.1:
            print("WARNING: First and second TF deviate more then 10 percent")
        meanTF=statistics.mean(TF)

    if len(peaks)>2 and len(valleys)>2:
        TF3=(peaks[2]-valleys[2])/valleys[2]
        TF.append(TF3)
        sorted_TF=sorted(TF)
    
        deviation_HL_TF=2*(sorted_TF[-1]-sorted_TF[0])/(sorted_TF[-1]+sorted_TF[0])
    
        if abs(deviation_HL_TF)> 0.10:
            deveation_sorted_1_2=2*(sorted_TF[0]-sorted_TF[1])/(sorted_TF[0]+sorted_TF[1])
            deveation_sorted_2_3=2*(sorted_TF[1]-sorted_TF[2])/(sorted_TF[1]+sorted_TF[2])
            if abs(deveation_sorted_1_2)<0.10:
                meanTF=statistics.mean(sorted_TF[0:1])
                print('WARNING: Highest TF excluded')
            elif abs(deveation_sorted_2_3)<0.10:
                meanTF=statistics.mean(sorted_TF[1:2])
                print('WARNING: Lowest TF excluded')
            else: 
                print('All three TFs differ more then 10 percent from each other')
                meanTF=statistics.mean(TF)
        

    print('All TFs:',TF)
        
    print('The mean TF is:', meanTF)
    
    win = hide_close_button(plt.gcf())
    coordinates = plt.ginput(total_clicks, timeout = 0, mouse_add= MouseButton.RIGHT, mouse_pop= MouseButton.MIDDLE, mouse_stop= MouseButton.BACK)
    plt.close("all")

    plt.show()

    

    if  'indexp3' in locals():
        indexp1=np.array(indexp1)
        indexp2=np.array(indexp2)
        indexp3=np.array(indexp3)
        indexv1=np.array(indexv1)
        indexv2=np.array(indexv2)
        indexv3=np.array(indexv3)

        indexp1=np.atleast_1d(indexp1)
        indexp2=np.atleast_1d(indexp2)
        indexp3=np.atleast_1d(indexp3)
        indexv1=np.atleast_1d(indexv1)
        indexv2=np.atleast_1d(indexv2)
        indexv3=np.atleast_1d(indexv3)


        indices_peaks=[indexp1[0],indexp2[0], indexp3[0]]
        indices_valleys=[indexv1[0],indexv2[0],indexv3[0]]
    elif 'indexp2' in locals(): 
        indexp1=np.array(indexp1)
        indexp2=np.array(indexp2)
        indexv1=np.array(indexv1)
        indexv2=np.array(indexv2)

        indexp1=np.atleast_1d(indexp1)
        indexp2=np.atleast_1d(indexp2)
        indexv1=np.atleast_1d(indexv1)
        indexv2=np.atleast_1d(indexv2)

        indices_peaks=[indexp1[0],indexp2[0]]
        indices_valleys=[indexv1[0],indexv2[0]]
    else:
        indexp1=np.array(indexp1)
        indexv1=np.array(indexv1)

        indexp1=np.atleast_1d(indexp1)
        indexv1=np.atleast_1d(indexv1)

        indices_peaks=[indexp1[0]]
        indices_valleys=[indexv1[0]]



            
    return peaks, valleys, TF, meanTF, indices_peaks, indices_valleys


def save(filename,peaks,valleys,TF,meanTF,indices_peaks, indices_valleys):
    


    time_axis=globals.time_axis_cal
    distances=globals.distance_in_mm_cal

    # Creating a new Excel workbook and renaming the first and only sheet
    wb = op.Workbook()
    sheet = wb.active
    sheet.title = "Overview"

    sheet.append(["Time (s)", "Thickness (mm)", "Peak","Valley","TFs","","","Mean TF"])


                
    for i in range(len(distances)):
    
        if i==indices_peaks[0]:
                if  i == 0:
                    sheet.append([time_axis[i], distances[i],peaks[0]],TF,meanTF)
                else:
                    sheet.append([time_axis[i], distances[i],peaks[0]])
        elif len(indices_peaks)>1 and i==indices_peaks[1]:
                if  i == 0:
                    sheet.append([time_axis[i], distances[i],peaks[1]],TF,meanTF)
                else:
                    sheet.append([time_axis[i], distances[i],peaks[1]])
        elif len(indices_peaks)>2 and i==indices_peaks[2]:
                if  i == 0:
                    sheet.append([time_axis[i], distances[i],peaks[2]],TF,meanTF)
                else:
                    sheet.append([time_axis[i], distances[i],peaks[2]])
        elif i==indices_valleys[0]:
            if  i == 0:
                sheet.append([time_axis[i], distances[i],"",valleys[0],TF,meanTF])
            else:
                sheet.append([time_axis[i], distances[i],"",valleys[0]])
        elif len(indices_valleys)>1 and i==indices_valleys[1]:
            if  i == 0:
                sheet.append([time_axis[i], distances[i],"",valleys[1],TF,meanTF])
            else:
                sheet.append([time_axis[i], distances[i],"",valleys[1]])

        elif len(indices_valleys)>2 and i==indices_valleys[2]:
            if  i == 0:
                sheet.append([time_axis[i], distances[i],"",valleys[2],TF,meanTF])
            else:
                sheet.append([time_axis[i], distances[i],"",valleys[2]])

        else: 
            if  i == 0:
                if len(TF)==1:
                    sheet.append([time_axis[i], distances[i],"","",TF[0],"","" ,meanTF])
                elif len (TF)==2:
                    sheet.append([time_axis[i], distances[i],"","",TF[0],TF[1],"" ,meanTF])
                elif len(TF)==3:
                    sheet.append([time_axis[i], distances[i],"","",TF[0],TF[1],TF[2],meanTF])
            else:
                sheet.append([time_axis[i], distances[i]])


            

        
    
    # Change decimals in value columns to 2 decimals
    for row in sheet.iter_rows(min_row=2, min_col=1, max_col=5):
        for cell in row:
            cell.number_format = '#,##0.00'
    
    # Make graph
    chart = ScatterChart()
    chart.title = "Distance-time"
    chart.x_axis.title = 'Time (s)'
    chart.y_axis.title = 'Thinkness (mm)'

    # Scale graph
    chart.x_axis.scaling.min = 0
    chart.x_axis.scaling.max = globals.time_axis_cal[-1]+1

    chart.y_axis.scaling.min = np.min(globals.distance_in_mm_cal)-0.2
    chart.y_axis.scaling.max = np.max(globals.distance_in_mm_cal)+0.2

    # Plot distance_time
    x_ref = Reference(sheet, min_col=1, max_col = 1, min_row = 2, max_row = len(globals.distance_in_mm_cal) + 1)
    y_ref = Reference(sheet, min_col=2, max_col = 2, min_row = 1, max_row = len(globals.distance_in_mm_cal) + 1) # min_row seems like a bug in lib
    series = Series(y_ref, x_ref, title_from_data = True)
    chart.series.append(series)

    # Plot peaks
    x_ref = Reference(sheet, min_col=1, max_col = 1, min_row = 2, max_row = len(globals.distance_in_mm_cal) + 1)
    y_ref = Reference(sheet, min_col=3, max_col = 3, min_row = 1, max_row = len(globals.distance_in_mm_cal) + 1) # min_row seems like a bug in lib
    series = Series(y_ref, x_ref, title_from_data = True)
    series.graphicalProperties.line.noFill = True
    series.marker = Marker(
        'circle',
        size=5,
        spPr=GraphicalProperties(solidFill="800000"))
    chart.series.append(series)

    # Plot valleys
    x_ref = Reference(sheet, min_col=1, max_col = 1, min_row = 2, max_row = len(globals.distance_in_mm_cal) + 1)
    y_ref = Reference(sheet, min_col=4, max_col = 4, min_row = 1, max_row = len(globals.distance_in_mm_cal) + 1) # min_row seems like a bug in lib
    series = Series(y_ref, x_ref, title_from_data = True)
    series.graphicalProperties.line.noFill = True
    series.marker = Marker(
        'circle',
        size=5,
        spPr=GraphicalProperties(solidFill="800000"))
    chart.series.append(series)

    sheet.add_chart(chart, "G2")
            
            
    wb.save(".".join(filename.split(".")[:-1]) + "-output.xlsx")