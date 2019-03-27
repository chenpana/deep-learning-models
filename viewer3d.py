#coding:utf-8


import sys
import time
import numpy as np
import vtk
import nibabel as nib
import nilearn.image
import scipy.ndimage as ndimage


# %%


def label_to_importer(d):
    """
    The numpy array of label send to vtk must be a cube shape
    I can not figure out why???
    """
    labels = np.unique(d)[1:]
    binarys=[]
    for label in labels:
        data=d.copy()
        data[data!=label]=0
        data[data==label]=1

        data_importer = vtk.vtkImageImport()

        data_importer.CopyImportVoidPointer(data,data.nbytes)

        data_importer.SetDataScalarTypeToUnsignedChar()
        data_importer.SetNumberOfScalarComponents(1)
        data_importer.SetDataExtent(0, data.shape[0]-1, 0, data.shape[1]-1, 0, data.shape[2]-1)
        data_importer.SetWholeExtent(0, data.shape[0]-1, 0, data.shape[1]-1, 0, data.shape[2]-1)
        binarys.append(data_importer)

    return binarys,labels


# %%


def reconstruct(readers, iso=1):
    extractors=[]
    for reader in readers:
        extractor = vtk.vtkMarchingCubes()
        extractor.SetInputConnection(reader.GetOutputPort())
        extractor.SetValue(0, iso)
        extractors.append(extractor)
    return extractors

def get_tableau_color():
    import matplotlib.colors as colors
    tableau=tuple(color for name, color in colors.TABLEAU_COLORS.items())
    surface_colors = vtk.vtkNamedColors()
    for i,item in enumerate(tableau):
        color=[int(item[1:3],16),int(item[3:5],16),int(item[5:7],16),255]
        surface_colors.SetColor("{}".format(i), color)
    return surface_colors

class RenderWindow():
    def __init__(self):
        aRenderer = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(aRenderer)

        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)         
      
        aCamera = vtk.vtkCamera()
        aCamera.SetViewUp(0, 0, -1)
        aCamera.SetPosition(0, -1, 0)
        aCamera.SetFocalPoint(0, 0, 0)
        aCamera.ComputeViewPlaneNormal()
        aCamera.Azimuth(30.0)
        aCamera.Elevation(30.0)

        # Actors are added to the renderer. An initial camera view is created.
        # The Dolly() method moves the camera towards the FocalPoint,
        # thereby enlarging the image.

        aRenderer.SetActiveCamera(aCamera)
        aRenderer.ResetCamera()
        aCamera.Dolly(1.5)

        # Set a background color for the renderer and set the size of the
        # render window (expressed in pixels).
        #aRenderer.SetBackground(colors.GetColor3d("BkgColor"))
        renWin.SetSize(640, 480)

        # Note that when camera movement occurs (as it does in the Dolly()
        # method), the clipping planes often need adjusting. Clipping planes
        # consist of two planes: near and far along the view direction. The
        # near plane clips out objects in front of the plane the far plane
        # clips out objects behind the plane. This way only what is drawn
        # between the planes is actually rendered.
        aRenderer.ResetCameraClippingRange()

        # Initialize the event loop and then start it.
        #iren.Initialize()
        #iren.Start() 
        
        self.aRenderer=aRenderer
        self.renWin=renWin
        self.iren=iren
        self.aCamera=aCamera    
    def add_actors(self,colors):
        aRenderer = self.aRenderer
        iren = self.iren

        for i, extractor in enumerate(extractors):
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(extractor.GetOutputPort())
            mapper.ScalarVisibilityOff()

            surface = vtk.vtkActor()
            surface.SetMapper(mapper)
            surface.GetProperty().SetDiffuseColor(colors.GetColor3d("{}".format(i)))
            aRenderer.AddActor(surface) 
        aRenderer.ResetCameraClippingRange()

        # Initialize the event loop and then start it.
        iren.Initialize()
        iren.Start()    
        
def surface_show(extractors,colors):
    
    aRenderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(aRenderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    

    for i, extractor in enumerate(extractors):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(extractor.GetOutputPort())
        mapper.ScalarVisibilityOff()

        surface = vtk.vtkActor()
        surface.SetMapper(mapper)
        surface.GetProperty().SetDiffuseColor(colors.GetColor3d("{}".format(i)))
        aRenderer.AddActor(surface)
        
    # It is convenient to create an initial view of the data. The FocalPoint
    # and Position form a vector direction. Later on (ResetCamera() method)
    # this vector is used to position the camera to look at the data in
    # this direction.
    aCamera = vtk.vtkCamera()
    aCamera.SetViewUp(0, 0, -1)
    aCamera.SetPosition(0, -1, 0)
    aCamera.SetFocalPoint(0, 0, 0)
    aCamera.ComputeViewPlaneNormal()
    aCamera.Azimuth(30.0)
    aCamera.Elevation(30.0)

    # Actors are added to the renderer. An initial camera view is created.
    # The Dolly() method moves the camera towards the FocalPoint,
    # thereby enlarging the image.

    aRenderer.SetActiveCamera(aCamera)
    aRenderer.ResetCamera()
    aCamera.Dolly(1.5)

    # Set a background color for the renderer and set the size of the
    # render window (expressed in pixels).
    #aRenderer.SetBackground(colors.GetColor3d("BkgColor"))
    renWin.SetSize(640, 480)

    # Note that when camera movement occurs (as it does in the Dolly()
    # method), the clipping planes often need adjusting. Clipping planes
    # consist of two planes: near and far along the view direction. The
    # near plane clips out objects in front of the plane the far plane
    # clips out objects behind the plane. This way only what is drawn
    # between the planes is actually rendered.
    aRenderer.ResetCameraClippingRange()

    # Initialize the event loop and then start it.
    iren.Initialize()
    iren.Start()   
    


# %%


def resample_to_cube(data):   
    shape = data.shape   
    new_shape=(np.max(shape),)*3
    new_data=np.zeros(new_shape,dtype=data.dtype)
    #slicer = [slice(int(round((new_shape[i]-shape[i])/2)),int(round((new_shape[i]+shape[i])/2))) for i in range(3)]
    slicer = [slice((new_shape[i]-shape[i])//2,(new_shape[i]+shape[i])//2) for i in range(3)]

    new_data[slicer[0],slicer[1],slicer[2]]=data
    return new_data


# %%


def show_nii_label_file(input_files, compatability=False):
    if not isinstance(input_files,(tuple,list)):
        input_files=(input_files,)
    if len(input_files) < 0:
        print("empty input file. just abort!")
        return
    
    nii_list = []
    slicer_list =[]
    for file in input_files:
        if isinstance(file,str):
            data_nii=nib.load(file)
        elif isinstance(file,(nib.Nifti1Image,nib.Nifti2Image)):
            data_nii=file
        else:
            print("find unkown item: {}".format(file))
            raise TypeError
            
        data_nii=nilearn.image.reorder_img(data_nii)
        nii_list.append(data_nii)
        slicer_list.append(ndimage.find_objects(data_nii.get_data()))
    
    # compare and show in a window label by label
    if compatability:
        extractors_list = []
        for data_nii in nii_list:
            data=data_nii.get_data()
            data=resample_to_cube(data)
            data_importers,labels=label_to_importer(data)
            extractors=reconstruct(data_importers)
            extractors_list.append(extractors)
        extractors_list=np.transpose(extractors_list)
        
        labels=np.unique(nii_list[0].get_data())[1:]
        for i,extractors in enumerate(extractors_list):
            print("show label: {}".format(labels[i]))
            surface_show(extractors,get_tableau_color())  
        return
    
    # show all label in a window for one data
    for data_nii in nii_list:
        data=data_nii.get_data()
        data=resample_to_cube(data)
        data_importers,labels=label_to_importer(data)
        print("find labels: ",labels)
        extractors=reconstruct(data_importers)
        surface_show(extractors,get_tableau_color())    



# %%

if __name__ == '__main__':
    """
    show a multiple label file
    """
    organs = [r"\esophagus_6222", r"\heart_2822", r"\trachea_2282", r"\aorta_2226"]
    index = r"\Patient_46"
    n = 2
    if n == 0:
        input_file = (r"C:\Users\陈攀\Desktop\complete\lixiaoying\食道后处理\post(1)\post"+index+ r".nii")
        show_nii_label_file(input_file)
    if n == 1:
        input_file = (r"C:\Users\陈攀\Desktop\viewer\submission_multi_scale_weighted\merge_45"+index+r".nii")
        show_nii_label_file(input_file)
    if n == 2:
        input_file = (
            r"C:\Users\陈攀\Desktop\complete\lixiaoying\食道后处理\post(1)\post"+index+ r".nii",# bule
                                r"C:\Users\陈攀\Desktop\viewer\submission\submission_4"+index+r".nii",  #brown
                              )
        show_nii_label_file(input_file,True)


# if __name__ == '__main__':
#     """
#     show a multiple label file
#     """
#     input_file = (r"C:\Users\陈攀\Desktop\viewer\submission_multi_scale_weighted\submission\Patient_47.nii")
#     show_nii_label_file(input_file)


# if __name__ == '__main__':
#     """
#     show a multiple label file
#     """
#     input_file = (r"C:\Users\陈攀\Desktop\viewer\train\Patient_01\truth.nii.gz")
#     show_nii_label_file(input_file)


# %%


# if __name__ == '__main__':
#     """
#     show two multiple label files by comparing label by label.
#     """
#     input_file = (r"C:\Users\陈攀\Desktop\viewer\submission_multi_scale_weighted\heart_2822\second_postprocess"+index+r"\prediction.nii.gz", #bule
#                         r"C:\Users\陈攀\Desktop\viewer\submission\submission_4\merge"+index+r"\prediction.nii.gz",  #brown
#                       )
#
#     show_nii_label_file(input_file,True)


# %%


# if __name__ == '__main__':

#     """
#     Show all label 3d reconstruction
#     """
#     #data_nii=nib.load("../segthor/training_clipped_all/prediction/Patient_25/prediction.nii.gz")
#     #data_nii=nib.load("../segthor/training_clipped_all/prediction/Patient_25/truth.nii.gz")
#     #data_nii=nib.load("../segthor/data/original/train/Patient_39/GT.nii.gz")
#     data_nii=nib.load("../segthor/test_heart/second_prediction/Patient_04/prediction.nii.gz")

#     #data_nii=nib.load("../segthor/test/second_preprocess/Patient_39/GT.nii.gz")
#     data_nii=nilearn.image.reorder_img(data_nii)


# %%


# if __name__ == '__main__':
#     """
#     The label send to vtk must be a cube shape
#     I can not figure out why???
#     """
#     data=data_nii.get_data()
#     data=resample_to_cube(data)
#     data_importers,labels=label_to_importer(data)
#     print("find labels: ",labels)
#     extractors=reconstruct(data_importers)
#     surface_show(extractors,get_tableau_color())


# %%


# if __name__ == '__main__':

#     """
#     Compare truth and prediction
#     """
#     truth_nii=nib.load("../segthor/valid_test/original//Patient_37/GT.nii.gz")
    
#     """
#     zoom to cube shape
#     """
#     truth=truth_nii.get_data()
#     truth=resample_to_cube(truth)

#     prediction_nii=nib.load("../segthor/valid_test/second_postprocess//Patient_37/prediction.nii.gz")
    
#     """
#     zoom to cube shape
#     """
#     prediction=prediction_nii.get_data()
#     prediction=resample_to_cube(prediction)

#     truth_importers,truth_labels=label_to_importer(truth)
#     truth_extractors=reconstruct(truth_importers)
#     print("truth label: ",truth_labels)


#     prediction_importers,prediction_labels=label_to_importer(prediction)
#     prediction_extractors=reconstruct(prediction_importers)
#     print("prediction label: ",prediction_labels)


# if __name__ == '__main__':

#     indice=1
#     surface_show([truth_extractors[indice],prediction_extractors[indice]],get_tableau_color())


# %%


## vtk data array to numpy
# import matplotlib.pyplot as plt
# from vtk.util.numpy_support import vtk_to_numpy
# vtk_data=data.copy()
# for i,reader in enumerate(data_importers):
#     reader.Update()
#     vtk_array = reader.GetOutput().GetPointData().GetArray(0)
#     vtk_array = vtk_to_numpy(vtk_array)
#     vtk_array=np.reshape(vtk_array,vtk_data.shape)
#     vtk_data[vtk_array==1]=i+1

