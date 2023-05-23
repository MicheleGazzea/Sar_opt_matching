import numpy as np
import tensorflow as tf

import tkinter as tk
from PIL import ImageTk
from PIL import Image, ImageOps, ImageDraw
from matplotlib import cm




class ImageGui_opt_sar(tk.Tk):
    """
    GUI for visualizing the SAR-opt dataset interactively.
    """
    
    def __init__(self, master, opt, sar, offset, heatmaps = None, feature_maps = None):
        """
        Initialise GUI
        """   
        
        # So we can quit the window from within the functions
        self.master = master
        
        # Extract the frame so we can draw stuff on it
        frame = tk.Frame(master)

        # Initialise grid
        frame.grid()
        # Start at the first file name
        assert opt.shape[0] == sar.shape[0] 
        self.index = 0
        self.n_images = opt.shape[0]
        
        self.opt = opt
        self.sar = sar
        self.offset = offset
        self.heatmaps = heatmaps
        self.feature_maps = feature_maps
        
        
        # Set empty image container
        self.image = None
        self.image2 = None
        self.image3 = None
        
        if not self.heatmaps is None:
            self.image4 = None
            
        if not self.feature_maps is None:
            self.image5 = None
            self.image6 = None
            
            if len(self.feature_maps) == 4:
                # it means we have multiscale feature too, we need to add two more containers
                self.image7 = None
                self.image8 = None
                
                
        # Set image panels
        self.image_panel = tk.Label(frame)
        self.image_panel2 = tk.Label(frame)
        self.image_panel3 = tk.Label(frame)
        
        if not self.heatmaps is None:
            self.image_panel4= tk.Label(frame)
        
        if not self.feature_maps is None:
            self.image_panel5 = tk.Label(frame)
            self.image_panel6 = tk.Label(frame)
            
            if len(self.feature_maps) == 4:
                # it means we have multiscale feature too, we need to add two more containers
                self.image_panel7 = tk.Label(frame)
                self.image_panel8 = tk.Label(frame)
                
        # Set image container to first image
        self.set_image()
            
        self.buttons = [] 
                            
        ### added in version 2
        self.buttons.append(tk.Button(frame, text="prev im 50", width=10, height=2, fg="purple", command=lambda l=0: self.show_prev_image50()))
        self.buttons.append(tk.Button(frame, text="prev im", width=10, height=2, fg="purple", command=lambda l=0: self.show_prev_image()))
        self.buttons.append(tk.Button(frame, text="next im", width=10, height=2, fg='purple', command=lambda l=0: self.show_next_image()))
        self.buttons.append(tk.Button(frame, text="next im 50", width=10, height=2, fg='purple', command=lambda l=0: self.show_next_image50()))
        # Save button
        self.buttons.append(tk.Button(frame, text="SAVE", width=10, height=2, fg='green', command=lambda l=0: self.save()))
        
        # Add progress label
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label = tk.Label(frame, text=progress_string, width=10)  
        self.progress_label.grid(row=2, column=3, sticky='we')
                                                                   
        # Place buttons in grid
        for ll, button in enumerate(self.buttons):
            button.grid(row=0, column=ll, sticky='we')
            frame.grid_columnconfigure(ll, weight=1)        
        
        
        # Place the image in grid
        
        # Optical
        self.image_panel.grid(row = 1, column = 0, sticky='we')
        # SAR
        self.image_panel2.grid(row = 1, column = 1, sticky='we')
        
        # Optical-SAR
        self.image_panel3.grid(row = 1, column = 2, sticky='we')
        if not self.heatmaps is None:
            # Heatmaps
            self.image_panel4.grid(row = 1, column = 3, sticky='we')
        if not self.feature_maps is None:
            # Feature map optical (original resolution)
            self.image_panel5.grid(row = 2, column = 0, sticky='we')
            # Feature map SAR (original resolution)
            self.image_panel6.grid(row = 2, column = 1, sticky='we')
            if len(self.feature_maps) == 4:
                # Feature map optical (downscaled)
                self.image_panel7.grid(row = 3, column = 0, sticky='we')
                # Feature map SAR (downscaled)
                self.image_panel8.grid(row = 3, column = 1, sticky='we')
        
                
#******************************************************************************        
    
    def numpy2PIL(self, data : np.array):
        """ Convert a np.array into a imagePIL
        'data' should be (N, H, W, Chs) """
        
        if len(data.shape) == 3:
            imagePIL = Image.fromarray( data[self.index, :, :] )
        
        elif len(data.shape) == 4:
            if data.shape[-1] == 1:
                imagePIL = Image.fromarray( data[self.index, :, :, 0] )
            else:
                imagePIL = Image.fromarray( data[self.index, :, :, :] )
        
        else:
            raise("Array has dimensions: " + str(data.shape) + ". Check dimensions")
        return imagePIL
    

    def save(self):
        # Save opt superimposed with SAR
        imagePIL_opt = self.numpy2PIL(self.opt)        
        imagePIL_sar = self.numpy2PIL(self.sar)
        
        row_true, col_true = np.unravel_index(np.argmax(self.offset[self.index, :,:]), shape = (65,65))
        imagePIL_opt.paste(imagePIL_sar, (col_true, row_true) )
        imagePIL_opt.save("visual_maps/" + str(self.index) + "_SAR_opt.png")
        
        if not self.heatmaps is None:
            heatmap = self.heatmaps[self.index,:,:]
            heatmap_PIL = Image.fromarray(cm.viridis(heatmap,  bytes=True))
            # Draw true match
            draw = ImageDraw.Draw(heatmap_PIL)
            draw.ellipse((col_true, row_true, (col_true+2), (row_true+2)), fill=(255, 0, 0), outline=(0, 0, 0))
            heatmap_PIL.save("visual_maps/" + str(self.index) + "_heatmap.png")
            
        if not self.feature_maps is None:
            psi_opt_o = self.feature_maps[0][self.index, :,:]
            psi_opt_o_PIL = Image.fromarray(cm.jet(psi_opt_o,  bytes=True))
            psi_opt_o_PIL.save("visual_maps/" + str(self.index) + "_psi_opt_o.png")
            
            psi_sar_o = self.feature_maps[1][self.index, :,:]
            psi_sar_o_PIL = Image.fromarray(cm.jet(psi_sar_o,  bytes=True))
            psi_sar_o_PIL.save("visual_maps/" + str(self.index) + "_psi_sar_o.png")
            
            if len(self.feature_maps) == 4:
                # Feature map optical (downscaled)
                psi_opt_d = self.feature_maps[2][self.index, :,:]
                psi_opt_d_PIL = Image.fromarray(cm.jet(psi_opt_d,  bytes=True))
                psi_opt_d_PIL.save("visual_maps/" + str(self.index) + "_psi_opt_d.png")
                
                # Feature map SAR (downscaled)
                psi_sar_d = self.feature_maps[3][self.index, :,:]
                psi_sar_d_PIL = Image.fromarray(cm.jet(psi_sar_d,  bytes=True))
                psi_sar_d_PIL.save("visual_maps/" + str(self.index) + "_psi_sar_d.png")

            
    def set_image(self):
        """
        Helper function which sets a new image in the image view
        """
        
        # 1-Optical:
        imagePIL_opt = self.numpy2PIL(self.opt).resize((256,256), Image.NEAREST) 
        self.image = ImageTk.PhotoImage(imagePIL_opt, master = self.master)
        self.image_panel.configure(image=self.image) 
        
        
        # 2-SAR: Grayscale
        imagePIL_sar = self.numpy2PIL(self.sar).resize((192,192), Image.NEAREST) 
        self.image2 = ImageTk.PhotoImage(imagePIL_sar, master = self.master)
        self.image_panel2.configure(image=self.image2)
        
        # 3-Optical super-imposed with SAR 
        row_true, col_true = np.unravel_index(np.argmax(self.offset[self.index, :,:]), shape = (65,65))
        imagePIL_opt.paste(imagePIL_sar, (col_true, row_true) )
        self.image3 = ImageTk.PhotoImage(imagePIL_opt, master = self.master)
        self.image_panel3.configure(image=self.image3)
        
        
        # 4-Heamaps
        if not self.heatmaps is None:
            heatmap = self.heatmaps[self.index,:,:]
            row_pred, col_pred = np.unravel_index(np.argmax(heatmap), shape = (65,65))
            heatmapPIL = Image.fromarray(cm.viridis(heatmap,  bytes=True)).resize((65*2,65*2), Image.NEAREST)
            
            # Draw true match
            draw = ImageDraw.Draw(heatmapPIL)
            draw.ellipse((col_true*2, row_true*2, (col_true+2)*2, (row_true+2)*2), fill=(255, 0, 0), outline=(0, 0, 0))
            
            # Draw predicted match
            # draw.ellipse((col_pred*2, row_pred*2, (col_pred+2)*2, (row_pred+2)*2), fill=(255, 0, 255), outline=(0, 0, 0))
            
            self.image4 = ImageTk.PhotoImage(heatmapPIL, master = self.master)
            self.image_panel4.configure(image=self.image4)
            
            
        if not self.feature_maps is None:
            
            # Feature map optical (original resolution)
            psi_opt_o = self.feature_maps[0][self.index, :,:]
            psi_opt_o_PIL = Image.fromarray(cm.jet(psi_opt_o,  bytes=True)).resize((256,256), Image.NEAREST)
            self.image5 = ImageTk.PhotoImage(psi_opt_o_PIL, master = self.master) 
            self.image_panel5.configure(image=self.image5)
            
            # Feature map SAR (original resolution)
            psi_sar_o = self.feature_maps[1][self.index, :,:]
            psi_sar_o_PIL = Image.fromarray(cm.jet(psi_sar_o,  bytes=True)).resize((192,192), Image.NEAREST)
            self.image6 = ImageTk.PhotoImage(psi_sar_o_PIL, master = self.master) 
            self.image_panel6.configure(image=self.image6)
            
            if len(self.feature_maps) == 4:
                # Feature map optical (downscaled)
                psi_opt_d = self.feature_maps[2][self.index, :,:]
                psi_opt_d_PIL = Image.fromarray(cm.jet(psi_opt_d,  bytes=True)).resize((256,256), Image.NEAREST)
                self.image7 = ImageTk.PhotoImage(psi_opt_d_PIL, master = self.master) 
                self.image_panel7.configure(image=self.image7)
                
                # Feature map SAR (downscaled)
                psi_sar_d = self.feature_maps[3][self.index, :,:]
                psi_sar_d_PIL = Image.fromarray(cm.jet(psi_sar_d,  bytes=True)).resize((192,192), Image.NEAREST)
                self.image8 = ImageTk.PhotoImage(psi_sar_d_PIL, master = self.master) 
                self.image_panel8.configure(image=self.image8)
            
                          
        
    def show_prev_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index -= 1
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string)
        
        if self.index >= 0:
            self.set_image()
        else:
        #   self.master.quit()
            self.master.destroy()
            
    def show_prev_image50(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index -= 50
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string)
        
        if self.index >= 0:
            self.set_image()
        else:
        #   self.master.quit()
            self.master.destroy()

            
    def show_next_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 1
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string) 

        if self.index < self.n_images:
            self.set_image()
        else:
#            self.master.quit()
            self.master.destroy() 
            
    def show_next_image50(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 50
        progress_string = "%d/%d" % (self.index+1, self.n_images)
        self.progress_label.configure(text=progress_string) 
        
        if self.index < self.n_images:
            self.set_image()
        else:
        #            self.master.quit()
            self.master.destroy()             
 
            
 
def visualize_dataset_with_GUI(dataset : tf.data.Dataset, heatmaps : np.array = None, feature_maps = None): 
    """
    Visualize a dataset with a GUI:
        opt_image | sAR_image | SAR into opt (correct pos) | similarity heatmap (if provided)
    """
    
    def process_feature_map(feature_map):
        # Average along the last axis
        feature_map = np.mean(feature_map, axis = -1)
        
        # Normalize the feature maps to have 0 mean and 1 standard deviation        
        feature_map = (feature_map - np.mean(feature_map, axis = 0)) / np.std(feature_map, axis = 0) 
        
        # Scale so that each heatmap is to [0,255]               
        a_min = feature_map.min(axis = (1,2), keepdims = True)
        a_max = feature_map.max(axis = (1,2), keepdims = True)
        feature_map = (((feature_map - a_min) / (a_max - a_min)) * 255).astype('uint8')
        
        return feature_map

                     
    dataset = dataset.unbatch()
    opt = []; sar = []; offset = []
    #Iterate through the tf.dataset
    for element in dataset.as_numpy_iterator():
        opt.append(element[0][0])
        sar.append(element[0][1])
        offset.append(element[1])
    opt = (np.array(opt)*255).astype('uint8')
    sar = (np.array(sar)*255).astype('uint8')
    offset = np.array(offset)
    
    if heatmaps is not None:
        # Process heatmaps for visualization: scale so that each heatmap is to [0,255]
        a_min = heatmaps.min(axis = (1,2), keepdims = True)
        a_max = heatmaps.max(axis = (1,2), keepdims = True)
        heatmaps = (((heatmaps - a_min) / (a_max - a_min)) * 255).astype('uint8')
        
    if feature_maps is not None:
        # Process feature maps for visualization: scale so that each heatmap is to [0,255]
        feature_maps = list(map(process_feature_map, feature_maps))

        
    # Run GUI
    root = tk.Tk()
    GUI = ImageGui_opt_sar(root, opt, sar, offset, heatmaps, feature_maps)
    root.mainloop()  





































