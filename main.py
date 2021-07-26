import cv2
import cv2 as cv
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
from tkinter import messagebox
from PIL import ImageTk, Image
import json
import glob
import os
import numpy as np
from detection import *
import classes
import config as cfg
import distortion_calibration as dc


videoCapture = cv2.VideoCapture(cfg.video)#, cv.CAP_DSHOW)  # starting capture immediately
myCalibration = classes.calibration()

#Show Output
def start():
    isTrue, frame = videoCapture.read()
    frame = myCalibration.preprocess(frame)

    if myCalibration.scale is None:
        print("No scale available. Please calibrate first!")
        return

    image, money = start_bv(frame, myCalibration.scale)

    show_img(panelA, image)
    label_2_euro.config(text="2€: " +  str(money[0]) + "x")
    label_1_euro.config(text="1€: " +  str(money[1]) + "x")
    label_50_ct.config(text="50ct: " + str(money[2]) + "x")
    label_20_ct.config(text="20ct: " + str(money[3]) + "x")
    label_10_ct.config(text="10ct: " + str(money[4]) + "x")
    label_5_ct.config(text="5ct: "  +  str(money[5]) + "x")
    label_2_ct.config(text="2ct: "  +  str(money[6]) + "x")
    label_1_ct.config(text="1ct: "  +  str(money[7]) + "x")
    label_falsch.config(text="Fakemoney: " + str(money[8]) + "x")
    label_gesamt.config(text="Total: " + str(money[9]) + "€")

def calibrate():
    global d
    cali = Toplevel()
    cali.wm_title("Calibration")

    a = Label(cali, text="Do you want to use old calibration pictures or take new ones?")
    a.grid(row=0, column=0, columnspan=2)

    def delete_cali_imgs_h():   
        # Alte Kalibrierbilder löschen
        cali.destroy()
        start_button.config(state=DISABLED)
        dc.delete_calib_imgs()
        new_calib_h()

    def calib_old_pictures_h():
        global myCalibration
        cali.destroy()
        start_button.config(state=DISABLED)
        images = glob.glob('*.jpg')
        done = dc.start_dist_calib(images, myCalibration)
        if done == True:
            response = messagebox.askokcancel("Calibration", "Distortion Calibration done")
            if response == True:
                next_calibration()
        else:
            messagebox.showerror("Error", "Couldn't find any Chessboards\nPlease start the calibration again!")
            

    def new_calib_h():
        global d
        d = 1
        win = Toplevel()
        win.wm_title("Distortion Calibration")

        l = Label(win, text="With the first calibration, the image distortion caused by the camera is eliminated.")
        l.grid(row=0, column=0, columnspan=10)

        m = Label(win, text="Please make sure to take at least five pictures with a chessboard as shown below.")
        m.grid(row=1, column=0, columnspan=10)

        

        init_img = prepare_init_image(cfg.path_calib)
        panelA = Label(win, image = init_img)
        panelA.grid(row=2, column=0, columnspan=10)
        panelA.image = init_img

        
        def take_picture_h():
            # videoCapture = cv2.VideoCapture(cfg.video, cv.CAP_DSHOW)
            global videoCapture
            global d
            isTrue, cali_img = videoCapture.read()
            file_name = str(d) + ".jpg"
            cv.imwrite(file_name, cali_img)
            b3.config(state=NORMAL)
            d +=1
            show_img(panelA, cali_img)
            check_calib()
            #videoCapture.release()
            
            

        def start_distortion_calibration_h():
            global myCalibration
            images = glob.glob('*.jpg')
            done = dc.start_dist_calib(images, myCalibration)
            if done == True:
                response = messagebox.askokcancel("Calibration", "Distortion Calibration done")
                if response == True:
                    win.destroy()
                    next_calibration()
            else:
                messagebox.showerror("Error", "Couldn't find any Chessboards\nPlease start the calibration again!")
                win.destroy()


        def delete_last_picture_h():
            global d
            d -= 1
            os.remove(str(d)+".jpg")
            image_deleted = cv.imread(cfg.path_deleted)
            show_img(panelA, image_deleted)
            b3.config(state=DISABLED)
            check_calib()

        def check_calib():
            global d
            if d >=6:
                b2.config(state=NORMAL)
            else:
                b2.config(state=DISABLED)


        b1 = Button(win, text="Take picture", padx=15, pady= 3, command=take_picture_h)
        b1.grid(row=3, column=0)

        b2 = Button(win, text="Calibrate", state=DISABLED, padx=15, pady= 3,command=start_distortion_calibration_h)
        b2.grid(row=3, column=2)

        b3 = Button(win, text="Delete last picture", state=DISABLED, padx=15, pady= 3,command=delete_last_picture_h)
        b3.grid(row=3, column=1)



    del_button = Button(cali, text="New fisheye Calibration", command=delete_cali_imgs_h)
    del_button.grid(row=1, column=0, padx=10, pady=10)

    use_button = Button(cali, text="Use old grid pictures", command=calib_old_pictures_h)
    use_button.grid(row=1, column=1, padx=10, pady=10)
    
       
    def next_calibration():
        cali.destroy()
        cali2 = Toplevel()
        cali2.wm_title("Calibration")

        p = Label(cali2, text="With the second calibration the scaling factor of the camera is measured")
        p.grid(row=0, column=0, columnspan=10)

        q = Label(cali2, text="Please take a photo of the calibration square as shown below")
        q.grid(row=1, column=0, columnspan=10)

        init_img = prepare_init_image(cfg.path_square)
        panel_calib = Label(cali2, image = init_img)
        panel_calib.grid(row=2, column=0, columnspan=10)
        panel_calib.image = init_img
        global frame_quad
        frame_quad = init_img
        
        def take_picture_h():
            global frame_quad
            isTrue, frame_quad = videoCapture.read()
            frame_quad_prep = myCalibration.preprocess(frame_quad)
            show_img(panel_calib, frame_quad_prep, inter=cv.INTER_CUBIC)
            b2.config(state=NORMAL)

        def start_quad_calibration_h():
            global frame_quad
            print("Calibrating...", end="")
            frame_quad_prep = myCalibration.preprocess(frame_quad)
            validationImg = myCalibration.calibration(frame_quad_prep, cfg.calibrateSquareSize, cfg.calibrationType)  # do calibration
            cv.imshow("Calibrated!", validationImg)
            cv.waitKey(300)
            print("Done!")
            response = messagebox.askokcancel("Calibration", "Calibration completed!")
            if response == True:
                cali2.destroy()
                cv.destroyAllWindows()
                start_button.config(state=NORMAL)


        b1 = Button(cali2, text="Take picture", padx=15, pady= 3, command=take_picture_h)
        b1.grid(row=3, column=0)

        b2 = Button(cali2, text="Calibrate", padx=15, pady= 3,command=start_quad_calibration_h)
        b2.grid(row=3, column=1)
        b2.config(state=DISABLED)

    skip_button = Button(cali, text="No fisheye calibration", command=next_calibration)
    skip_button.grid(row=1, column=2, padx=10, pady=10)



def save_calibration(window):
    """Save a calibration file."""
    window.destroy()

    
    # cam_matrix = np.loadtxt("camMatrix.csv", delimiter=',')
    # cam_matrix = cam_matrix.tolist()
    # dist_param = np.loadtxt("distParam.csv", delimiter=',')
    # dist_param = dist_param.tolist()

    # write important info into dict
    dic = {
        "scale": myCalibration.scale,
        "cornerPoints": myCalibration.cornerPoints,
        "cameraMatrix": myCalibration.cam_matrix,
        "distParam": myCalibration.dist_param
        #"cameraMatrix": cam_matrix,
        #"distParam": dist_param
    }
    print(dic)

    file = asksaveasfile(initialdir = "./", 
                         filetypes = [("Json files", "*.json")], 
                         defaultextension = [("Json files", "*.json")])

    if file is None: 
        print("Error: No file selected")
        return

    with open(file.name, 'w') as json_file:
        json.dump(dic, json_file)

    messagebox.showinfo("Save Calibration", "Successfully saved calibration!")
    print("Successfully saved calibration!")


def load_calibration(window):
    """load a calibration file."""
    window.destroy()

    file = filedialog.askopenfilename(initialdir = "./calibrations",
                                          title = "Select a File",
                                          filetypes = (("Json files", "*.json"),
                                                       ("all files", "*.*"))
                                        )
    if file is None or file == '': 
        print("Error: File not found!")
        return

    with open(file) as f:
        data = json.load(f)
    
    myCalibration.scale = data["scale"]
    myCalibration.cornerPoints = data["cornerPoints"]
    myCalibration.cam_matrix = data["cameraMatrix"]
    myCalibration.dist_param = data["distParam"]

    # # Previous fisheye calibration storage
    # cam_matrix = data["cameraMatrix"]
    # np.savetxt("camMatrix.csv", cam_matrix, delimiter=",")
    # dist = data["distParam"]
    # np.savetxt("distParam.csv", dist, delimiter=",")

    messagebox.showinfo("Load Calibration", "Successfully loaded calibration!")
    start_button.config(state=NORMAL)
    print("Successfully loaded calibration!")

def load_default(window):
    """load default calibration file."""
    window.destroy()

    file = cfg.default_config_path

    with open(file) as f:
        data = json.load(f)
    
    myCalibration.scale = data["scale"]
    myCalibration.cornerPoints = data["cornerPoints"]
    myCalibration.cam_matrix = data["cameraMatrix"]

    # np.savetxt("camMatrix.csv", cam_matrix, delimiter=",")
    myCalibration.dist_param = data["distParam"]
    # np.savetxt("distParam.csv", dist, delimiter=",")

    messagebox.showinfo("Load default Calibration", "Successfully loaded default calibration!")
    start_button.config(state=NORMAL)
    print("Successfully loaded default calibration!")
                                          

def save_and_load_popup():  # popup window
    win = Toplevel()
    win.wm_title("Save/Load")

    l = Label(win, text="Do you want to load or save the current configuration")
    l.grid(row=0, column=0, columnspan=4)

    def save_calibration_h():
        save_calibration(win)
    def load_calibration_h():
        load_calibration(win)
    def load_default_h():
        load_default(win)

    b1 = Button(win, text="Save", command=save_calibration_h)
    b1.grid(row=1, column=0, sticky='w', padx=10, pady=10)

    b2 = Button(win, text="Load", command=load_calibration_h)
    b2.grid(row=1, column=1, padx=10, pady=10)

    b4 = Button(win, text="Load Default", command=load_default_h)
    b4.grid(row=1, column=2, padx=10, pady=10)

    b3 = Button(win, text="Exit", command=win.destroy)
    b3.grid(row=1, column=3, sticky='e', padx=10, pady=10)



def show_img(panel, img, inter=cv.INTER_AREA):
    """Displays an opencv BGR-image into the desired tiker panel."""
    currImg = panel.image

    img = cv.resize(img.copy(), (IMGx, IMGy), interpolation=inter)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    tinkerFrame = Image.fromarray(img)
    tinkerFrame = ImageTk.PhotoImage(tinkerFrame)
    panel.configure(image=tinkerFrame)
    panel.image = tinkerFrame

#Open Initial Image
init_img = cv2.imread(cfg.path_main)
IMGy, IMGx, _ = init_img.shape
IMGy = round(IMGy * cfg.imagescale)
IMGx = round(IMGx * cfg.imagescale)
dimStr = str(IMGx + 4) + 'x' + str(IMGy + 160)

root = Tk()
root.title("Coin and Fakemoney Detection")
root.geometry(dimStr)
root.iconbitmap(cfg.path_icon)

# Display initial Image
init_img = cv.resize(init_img, (IMGx, IMGy), interpolation=cv.INTER_CUBIC)
init_img = cv.cvtColor(init_img, cv.COLOR_BGR2RGB)
init_img = Image.fromarray(init_img)
init_img = ImageTk.PhotoImage(init_img)
panelA = Label(image = init_img)
panelA.grid(row=1, column=0, columnspan=3)
panelA.image = init_img


start_button = Button(root, text="Start", state=DISABLED, padx=50, pady=10, command=start)
start_button.grid(row=0, column=0, columnspan=1)

calib_button = Button(root, text="Calibrate", padx=50, pady=10, command=calibrate)
calib_button.grid(row=0, column=1, columnspan=1)

calib_button = Button(root, text="Save/Load Calibration", padx=50, pady=10, command=save_and_load_popup)
calib_button.grid(row=0, column=2, columnspan=1)


label_1 = Label(text="Detected Coins:")
label_1.grid(row=2, column=0, columnspan=3)

label_2_euro = Label(text="2€:")
label_2_euro.grid(row=3, column=0)

label_1_euro = Label(text="1€:")
label_1_euro.grid(row=4, column=0)

label_50_ct = Label(text="50ct:")
label_50_ct.grid(row=5, column=0)

label_20_ct = Label(text="20ct:")
label_20_ct.grid(row=3, column=1)

label_10_ct = Label(text="10ct:")
label_10_ct.grid(row=4, column=1)

label_5_ct = Label(text="5ct:")
label_5_ct.grid(row=5, column=1)

label_2_ct = Label(text="2ct:")
label_2_ct.grid(row=3, column=2)

label_1_ct = Label(text="1ct:")
label_1_ct.grid(row=4, column=2)

label_falsch = Label(text="Fake Money:")
label_falsch.grid(row=5, column=2)

label_gesamt = Label(text="Total:")
label_gesamt.grid(row=6, column=0, columnspan=3)

def prepare_init_image(image_path):
    #Open Initial Image
    init_img = cv.imread(image_path)
    IMGy, IMGx, _ = init_img.shape
    IMGy = round(IMGy * cfg.imagescale)
    IMGx = round(IMGx * cfg.imagescale)

    # Display initial Image
    init_img = cv.resize(init_img, (IMGx, IMGy), interpolation=cv.INTER_CUBIC)
    init_img = cv.cvtColor(init_img, cv.COLOR_BGR2RGB)
    init_img = Image.fromarray(init_img)
    init_img = ImageTk.PhotoImage(init_img)
    return init_img


root.mainloop()
