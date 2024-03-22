from windows import get_echo_images, crop_and_canny, get_distance, get_TF, save
from pynput import keyboard, mouse
from helper import on_c_press



[filename,images,cine_rate, depth]=get_echo_images()

length_frame = 1 / cine_rate 

[images_masked,x,y1,y2]=crop_and_canny(images)

# keyboard_listener = keyboard.Listener(on_press=on_key_press)
keyboard_listener_c=keyboard.Listener(on_press=on_c_press)

# keyboard_listener.start()
keyboard_listener_c.start()
get_distance(images,images_masked,x,y1,y2, length_frame,depth)
keyboard_listener_c.stop()
# keyboard_listener.stop()

[peaks, valleys, TF, meanTF,indices_peaks, indices_valleys]=get_TF()

save(filename,peaks, valleys, TF, meanTF,indices_peaks, indices_valleys)

