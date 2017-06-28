import tensorflow as tf
import os.path

def getSubdirectories(parentDirectory):
	subDirs = [x[0] for x in tf.gfile.Walk(parentDirectory)]
	# The root directory comes first, so skip it.
	return subDirs[1:]

def getFilesFromDirectoriesWithExtension(directoryPaths,extensionList):
	fileListMap = {}
	for directoryPath in directoryPaths:
		fileList = []
		for extension in extensionList:
      			fileGlob = os.path.join(directoryPath, '*.' + extension)
	    		for file_name in tf.gfile.Glob(fileGlob):
      				file_base_name = os.path.basename(file_name)
				fileList.append(file_base_name)
		fileListMap[os.path.basename(directoryPath)] = fileList
	return fileListMap

def get_images_from_directory(directoryPath):
	extensions = ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG']
	fileList = getFilePathsFromDirectoryWithExtension(directoryPath,extensions)
	return fileList

def getFilePathsFromDirectoryWithExtension(directoryPath,extensionList):
	file_paths = []
	for extension in extensionList:
		fileGlob = os.path.join(directoryPath, '*.' + extension)
		file_paths.extend(tf.gfile.Glob(fileGlob))
	return file_paths

def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
