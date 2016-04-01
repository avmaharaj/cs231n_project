import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave




def my_load_tiny_imagenet(path, dtype=np.float32, discret = 256, npix=224):
  """
  Modified version of LoadTinyImageNet from CS231n

  Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
  TinyImageNet-200 have the same directory structure, so this can be used
  to load any of them.

  Inputs:
  - path: String giving path to the directory to load.
  - discret: the desired discretization of the data 
  - npix : the desired output size of the data
  - dtype: numpy datatype used to load the data.

  Returns (only) :
  - class_names: A list where class_names[i] is a list of strings giving the
    WordNet names for class i in the loaded dataset.

  Writes to disk:
  - training and validation datasets with same folder structure as original data
  - "val<discret>.txt" which contains paths to images and their image class
  - "train<discret>.txt" which contains path to training images and their class
  """
  # First load wnids
  #print discret
  #Set up some folder names that we will write to
  trainfolder = "train" + str(discret)
  valfolder = "val" + str(discret)
  trainfilename = trainfolder + ".txt"
  valfilename = valfolder + ".txt"


  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # Map wnids to integer labels
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # Use words.txt to get names for each class
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.iteritems():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]


 

  # First open the file to which we will write paths and labels
  trainfile = open(os.path.join(path,trainfilename),'w')

  # Next load training data.
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print 'loading training data for synset %d / %d' % (i + 1, len(wnids))
    
    # To figure out the filenames we need to open the boxes file
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)
    
   
    #create the directory to which we will write a new image
    os.makedirs(os.path.join(path, trainfolder, wnid, 'images'))

    #Now start looping over all photos in given directory
    for j, img_filename in enumerate(filenames):
      
      #read the image in
      img_file = os.path.join(path, 'train', wnid, 'images', img_filename)
      img = imread(img_file)
     
      #set up the name of the image, and appropriate paths
      nojpg_filename = img_filename.split('.')[0]
      png_filename = nojpg_filename + '.png'

      #resize the image by calling scipy's function
      img = imresize(img ,(npix,npix), 'bilinear')

      #deal with grayscale images by just coping input channels
      #THIS IS A HACK  
      if img.ndim == 2:
        tmp = np.zeros((3,npix,npix))
        tmp[0] = img
        tmp[1] = img
        tmp[2] = img
        img = tmp.transpose(1,2,0)

      #Now do the discretization
      img = np.floor_divide(img,(256/discret))*(256/discret) + 0.5*(256/discret)
      img = img.astype('uint8')

      #create the new image file's location and name
      new_img_file = os.path.join(path, trainfolder, wnid, 'images', png_filename)
      filepath = os.path.join(wnid, 'images', png_filename)
      
      #Save the image in the new location
      imsave(new_img_file, img)
      
      #write the path and label in the valfile 
      outdata = filepath + '\t' + str(wnid_to_label[wnid]) + '\n'
      trainfile.write(outdata)

      

  trainfile.close()      




  # Now move on the validation data
  valfile = open(os.path.join(path,valfilename),'w')
  #create the directory to which we will write a new image
  os.makedirs(os.path.join(path, valfolder,'images'))
  
  # Next load validation data
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    
    #Parse the val_annotations file for labels and images
    for line in f:
      img_file, wnid = line.split('\t')[:2]
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)
    
    #Store the image labels
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])


    #Now loop over images in image folder
    for i, img_filename in enumerate(img_files):
      if (i + 1) % 1000 == 0:
        print 'loading valdata %d / %d' % (i + 1,num_val)
      
      #set up some filepaths
      img_file = os.path.join(path, 'val', 'images', img_filename)
      nojpg_filename = img_filename.split('.')[0]
      png_filename = nojpg_filename + '.png'

      #Read the image
      img = imread(img_file)
      #Resize it
      img = imresize(img ,(npix,npix), 'bilinear')

      #Deal with grayscale (once more, a hack)
      if img.ndim == 2:
        tmp = np.zeros((3,npix,npix))
        tmp[0] = img
        tmp[1] = img
        tmp[2] = img
        img = tmp.transpose(1,2,0)

      #Do the discretization
      img = np.floor_divide(img,(256/discret))*(256/discret) + 0.5*(256.0/discret)
      img = img.astype('uint8')
      

      new_img_file = os.path.join(path, valfolder, 'images', png_filename)
      #Save the image in the new location
      imsave(new_img_file,img)
      
      #write the path and label in the valfile 
      outdata = png_filename + '\t' + str(y_val[i]) + '\n'
      valfile.write(outdata)
      # X_val[i] = img.transpose(2, 0, 1)

  valfile.close()

  return class_names







def load_tiny_imagenet_val(path, dtype=np.float32, discret = 256, npix=224):
  """
  Modified version of LoadTinyImageNet from CS231n

  Loads the validation images from  TinyImageNet. 

  Inputs:
  - path: String giving path to the directory to load.
  - discret: the desired discretization of the data 
  - npix : the desired output size of the data
  - dtype: numpy datatype used to load the data.

  Returns (only) :
  - class_names: A list where class_names[i] is a list of strings giving the
    WordNet names for class i in the loaded dataset.
  - X_val: an np.array of size (N, 3, npix, npix) where N is number of images
  - y_val: an array of size (N) with class labels


   """
  # First load wnids
  #Set up some folder names that we will write to
  
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # Map wnids to integer labels
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # Use words.txt to get names for each class
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.iteritems():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]


    
  # Next load validation data
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    
    #Parse the val_annotations file for labels and images
    for line in f:
      img_file, wnid = line.split('\t')[:2]
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)
    
    #Store the image labels
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    X_val = np.zeros((num_val, 3, npix, npix), dtype=dtype)
    # X_val_centered = np.zeros((num_val, 3, npix, npix), dtype=dtype)
    # diff = np.zeros((3,1,1))
    # diff[0] = 104
    # diff[1] = 117
    # diff[2] = 123

    #Now loop over images in image folder
    for i, img_filename in enumerate(img_files):
      if (i + 1) % 1000 == 0:
        print 'loading valdata %d / %d' % (i + 1,num_val)
      
      #set up some filepaths
      img_file = os.path.join(path, 'val', 'images', img_filename)
      nojpg_filename = img_filename.split('.')[0]
      png_filename = nojpg_filename + '.png'

      #Read the image
      img = imread(img_file)
      #Resize it
      img = imresize(img ,(npix,npix), 'bilinear')

      #Deal with grayscale (once more, a hack)
      if img.ndim == 2:
        tmp = np.zeros((3,npix,npix))
        tmp[0] = img
        tmp[1] = img
        tmp[2] = img
        img = tmp.transpose(1,2,0)

      #Do the discretization
      img = np.floor_divide(img,(256/discret))*(256/discret) + 0.5*(256.0/discret)
      img = img.astype('uint8')
      
      X_val[i] = img.transpose(2, 0, 1)
      # X_val_centered[i] = X_val[i] - diff



  return class_names, X_val, y_val


