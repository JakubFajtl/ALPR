import os
import glob
import cv2
import pandas as pd

class AnnotationTool:
    def __init__(self,validation_frames_folder="validationSetStuff/validation_frames",
                 output_csv="ground_truth_localization.csv"):
        self.validation_frames_folder = validation_frames_folder
        self.output_csv = output_csv
        #gets list of frame paths
        self.image_paths = sorted(glob.glob(os.path.join(validation_frames_folder, "*.jpg")))

        #state variables for drawing
        self.drawing = False
        self.ix, self.iy = -1, -1  # start coords = left top
        self.fx, self.fy = -1, -1  # end coords = right bottom
        self.current_img = None
        self.temp_img = None
        self.annotations = []  # list to store data
        self.gt_boxes = []

    # handles the mouse drawing logic - flags and param are needed for cv2
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                #draw on a copy to see box while drawing
                self.temp_img = self.current_img.copy()

                #redraw exisisting box
                for box in self.gt_boxes:
                    cv2.rectangle(self.temp_img, (box[2], box[0]), (box[3], box[1]), (0, 255, 0), 2)
                #draw active box
                cv2.rectangle(self.temp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Annotation Tool', self.temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            #draw box so we see it for confirmation
            self.temp_img = self.current_img.copy()
            #sort coords to ensure correct box dimensions before adding to list
            x1, x2 = sorted([self.ix, self.fx])
            y1, y2 = sorted([self.iy, self.fy])
            self.gt_boxes.append([y1, y2, x1, x2])
            self.update_display()

    #helper function to redraw all boxes already on image
    def update_display(self):
        self.temp_img = self.current_img.copy()
        for box in self.gt_boxes:
            cv2.rectangle(self.temp_img, (box[2], box[0]), (box[3], box[1]), (0, 255, 0), 2)
        cv2.imshow('Annotation Tool', self.temp_img)

    #main loop for annotation tool
    def run(self):
        if not self.image_paths:
            print(f"No images found in {self.validation_frames_folder}")
            return
        #setup window and drawing capabilities
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        print("Controls for annotator: [c] Confirm, [r] Reset, [s] Skip, [q] Quit & Save")
        # frames loop
        for image_path in self.image_paths:
            frame_filename = os.path.basename(image_path)
            try:
                frame_num = int(frame_filename.split("_")[1].split(".")[0]) #get frame number from file name
            except ValueError:
                frame_num = frame_filename

            self.current_img = cv2.imread(image_path)
            self.temp_img = self.current_img.copy()

            # reset values before annotation loop
            self.ix, self.iy, self.fx, self.fy = -1, -1, -1, -1
            self.gt_boxes = [] #for new frame boxes
            while True:
                #update only if not dragging mouse = drawing
                if not self.drawing:
                    cv2.imshow("Annotation Tool", self.temp_img)

                k = cv2.waitKey(1) & 0xFF

                if k == ord("c"): #confirm
                    if len(self.gt_boxes) > 0:
                        #save to annotation list
                        for box in self.gt_boxes:
                            self.annotations.append({"Frame Number": frame_num,
                                                 "Top": box[0], "Bottom": box[1], "Left": box[2], "Right": box[3]})
                            print(f"Saved annotation for frame {frame_num}")
                    else:
                        print(f"No valid box selected for frame {frame_num}")
                        #save as 0s if no valid box since frame has no plate - will allow for easier evaluation
                        self.annotations.append({"Frame Number": frame_num,
                                                 "Top": 0, "Bottom": 0, "Left": 0, "Right": 0})
                    break

                elif k == ord("r"): #reset
                    self.ix, self.iy, self.fx, self.fy = -1, -1, -1, -1
                    self.gt_boxes = []
                    self.temp_img = self.current_img.copy()
                    print(f"reset selection")

                elif k == ord("s"): #skip
                    print(f"Skipped frame {frame_num}")
                    break

                elif k == ord("q"): #quit
                    print("Quitting annotation tool")
                    self.save_csv()
                    cv2.destroyAllWindows()
                    return
        #after loop is done save annotations
        self.save_csv()
        cv2.destroyAllWindows()
        print("Done, closing...")


    #saving annotations to csv using panda, headers are from dictionary keys
    def save_csv(self):
        if not self.annotations:
            print("No annotations to save")
            return
        df = pd.DataFrame(self.annotations)
        df.to_csv(self.output_csv, index=False)
        print(f"Annotations saved to {self.output_csv}")

#safe main function for annotation tool
if __name__ == "__main__":
    tool = AnnotationTool()
    tool.run()
