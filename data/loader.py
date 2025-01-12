import gzip
import struct
import numpy as np

def images_loader(filename):
    """
    加载 MNIST 图像训练集。
    
    文件结构:
        0000     32 bit integer    0x00000803(2051) magic number

        0004     32 bit integer    10000 (实际数据集为 60000)            number of images

        0008     32 bit integer    28               number of rows

        0012     32 bit integer    28               number of columns

        0016     unsigned byte     ??               pixel (像素)

        0017     unsigned byte     ??               pixel (像素)
        
        ...
    返回格式：
        X_array_normalized: 
            2D: (imagesNum, rows * cols)
    """
    with gzip.open(filename, 'rb') as f:
        magicNum, imagesNum, rows, cols = struct.unpack('>4i', f.read(16))
   
        
        if magicNum != 0x00000803:
            raise ValueError(f'Error: magic number is not 0x00000803, but {magicNum}')
        
        unpack_data = struct.unpack(f'>{imagesNum * rows * cols}B', f.read())

        X_array = np.array(unpack_data, dtype=np.float32).reshape(imagesNum, rows * cols)

        # 归一化
        X_array_normalized = X_array / 255.0

        return X_array_normalized



def labels_loader(filename):
    """
    加载 MNIST 标签训练集。
    
    文件结构:
        0000     32 bit integer    0x00000801(2049) magic number (MSB first)
        
        0004     32 bit integer    60000            number of items
        
        0008     unsigned byte     ??               label
        
        0009     unsigned byte     ??               label
        
        ...
    返回格式：
        Y_array: 
            2D: (imagesNum, 1)
    """
    with gzip.open(filename, 'rb') as f:
        magicNum, labelsNum = struct.unpack('>2i', f.read(8))
        
        if magicNum != 0x00000801:
            raise ValueError(f'Error: magic number is not 0x00000801, but {magicNum}')
        
        unpack_data =  struct.unpack(f'>{labelsNum}B', f.read())

        Y_array = np.array(unpack_data, dtype=np.uint8).reshape(labelsNum,)
        return Y_array
    



def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    X = images_loader(image_filename)
    y = labels_loader(label_filename)

    return X, y