#pip install ultralytics
import os
import time

from ultralytics import YOLO
import sys
# Load a model
import sys, os
# import warning
# warnings.filterwarnings("ignore")
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    configString = sys.argv[1]

    #InstalledLocation##im$$imageLocation

    configs = configString.split("##im$$")

    modelPath = os.path.join(configs[0],"All21PII.pt")

    folderPath = configs[1]
    model = YOLO(modelPath)

    #"C:\Users\SISAManikChitralwar\Documents\ML Image\ExeWithFolder##im$$C:\Users\SISAManikChitralwar\Pictures\iMAGES"
    names = ['AadharCard', 'AmericanExpress', 'CanaraBankLOGO', 'DinersClub', 'Discover', 'DrivingLicense', 'IndianPassport', 'JCB', 'Maestro', 'MasterCard', 'NINO -UK-', 'NRIC -Singapore-', 'National ID -Saudi Arabia-', 'National ID -UAE-', 'PANCard', 'Rupay', 'SSN', 'UnionPay', 'VISA', 'VoterId']
    x = [os.path.join(r, file) for r, d, f in os.walk(folderPath) for file in f]
    for filePath in x:
        result = model(filePath)
        boxes = result[0].boxes
        Clsses = []
        for ele in boxes.cls:
            Clsses.append(names[int(ele)])

        confidences = []
        for con in boxes.conf:
            confidences.append(float(con))

        res = dict(zip(confidences, Clsses))
        if len(res) == 0:
            print("$*$" + "Class Not Found"+ "$*$")
        else:
            for key, value in res.items():
                print("$*$" + str(value)+ ':'+ str(key) + "$*$")