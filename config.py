
# Video device or file
video = 0  # number (0) for live webcam, or path for video
video = r"testvideos\Kalibrierung.mp4"
#video = r"testvideos\2.mp4"
#video = r"testvideos\TestKreise.mp4"
#video = r"testvideos\M1.mp4"
#video = r"testvideos\M_Kali.mp4"
#video = r"testvideos\4K.mp4"
#video = r"testvideos\4K_Kali.mp4"
#video = r"testvideos\G1.mp4"
video = r"testvideos\G2.mp4"
#video = r"testvideos\After Update 2.mp4"
video = r"testvideos\Euros_2_nahe.mp4"
# video = r"testvideos\schräg1.mp4"

# video = r"testvideos\fish and scale calibration.mp4"

#video = r"testvideos\Euros.mp4"

#video = r"testvideos\Euros_2_nahe.mp4"

imagescale = 0.7  # scale image (only for displaying)

frame_size = (1920, 1080) #camera resolution

coinsCanOverlap = False  # True, if coins can overlap



# Diameter EURO coins
coinDia = {
    "1c":  16.25,
    "2c":  18.75,
    "5c":  21.25,
    "10c": 19.75,
    "20c": 22.25,
    "50c": 24.25,
    "1e":  23.25,
    "2e":  25.75,
    "tolerance+-": 0.4,  # tolerance for still detecting as coin of this size (max 0.499)
    "unit": "mm"
}
# Values for circle detection
minDiameter = 14  # [mm]
maxDiameter = 29  # [mm]


# Color EURO coins ([Hueangle, tolerance])
coinColor = {
    "ct_red": [350, 25],
    "ct_yellow": [62, 9],
    "euro_yellow": [62, 9]
}


### Calibration
calibrationType = 'area'  # choose type 'area' (normal) or 'warp' (for a non orthogonal cameraangle)
#calibrateColor = (127, 127, 127)  # color of square
calibrateSquareSize = 100  # sidelength of square [mm]

#Chessboard size
chessboard_size = (17, 9) #column-1, row-1

warpscale = 0.9  # zoom after warping (>1: Zoom in; <1: Zoom out) | only for calibarationType 'warp'

# Background HUE
hueAngleMean = 156  # mean hue angle [0, 359]
hueTolerance = 6  # +/- tolerance e.g:  [angle - tolerance, angle + tolerance]


##### Filepaths #####
#default config
default_config_path = r".\calibrations\default.json"

#icon
path_icon = r".\resources\euro_coins.ico"

#initial image main window
path_main = r".\resources\defaultIMG.jpg"

#initial image scale calibration
path_square = r".\resources\calib_square.jpg"

#initial image distortion calibration
path_calib = r".\resources\Initial_image_calibration.jpg"

#image deleted
path_deleted = r".\resources\deleted.PNG"