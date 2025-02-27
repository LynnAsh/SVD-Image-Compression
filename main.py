from PIL import Image
import numpy as np
import os

imgFile = input('Image file name or path: ')

# gets and returns several k values to test and compare
def get_k_array() -> list | None:
    isValid = False

    while not isValid:
        k_array = input('Enter values of k to test (k1 k2 k3 ...): ')
        try:
            k_array = list(map(int, k_array.split(' ')))
            return k_array
        except ValueError:
            print('Must be integer')

# returns a numpy array from an image file
def get_image_array(image):
    image_array = np.array(image)
    return image_array

# returns the file size of an image file
def get_image_size(image) -> float:
    image_size = round((os.stat(image).st_size / 1000), 2)
    return image_size

def save_compressed_image(image, k) -> None:
    image = image.astype(np.uint8) # rounds off floats to ints
    image = Image.fromarray(image) # converts back to an image file to save
    image.save(f'comp{k}.jpg')



def compress_image_greyscale(image) -> None:
    image = image.convert('L') # converts image into greyscale

    kArr = get_k_array()
    imgArr = get_image_array(image)

    imgSizeOG = get_image_size(imgFile)
    print(f'------------------\nSize before compression: {imgSizeOG} kilobytes')

    U, S, V = np.linalg.svd(imgArr) # splits the pixel matrix into the 3 SVD matrices

    S = np.diag(S) # converts S into a diagonal matrix
    S_compressed = np.zeros((U.shape[0], V.shape[0])) # creates an empty matrix with the same dimensions as S

    for k in kArr:
        try:
            S_compressed[:k, :k] = S[:k, :k] # truncates S to a kxk matrix and assigns it to another kxk sized S_compressed
            compressedImg = U @ S_compressed @ V # multiplies the 3 SVD matrices back into a single matrix

            save_compressed_image(compressedImg, k)

            imgSizeCmp = get_image_size(f'comp{k}.jpg')
            cmpRatio = imgSizeOG / imgSizeCmp # calculates the compression ratio
            print(f'[k={k}] File Size: {imgSizeCmp} kilobytes | Compression Ratio: {round(cmpRatio, 4)}')

        except ValueError:
            print(f'k larger than singular values matrix (k={k})')

def compress_image_color(image) -> None:
    kArr = get_k_array()
    imgArr = get_image_array(image)

    imgSizeOG = get_image_size(imgFile)
    print(f'------------------\nSize before compression: {imgSizeOG} kilobytes')

    # splits the 3D image array into three 2D arrays for each color channel
    R = imgArr[:, :, 0]
    G = imgArr[:, :, 1]
    B = imgArr[:, :, 2]

    # splits the pixel matrices into the 3 SVD matrices
    UR, SR, VR = np.linalg.svd(R)
    UG, SG, VG = np.linalg.svd(G)
    UB, SB, VB = np.linalg.svd(B)

    # converts the S matrices into a diagonal matrix
    SR = np.diag(SR)
    SG = np.diag(SG)
    SB = np.diag(SB)

    # creates empty matrices with the same dimensions as the S matrices
    SR_compressed = np.zeros((UR.shape[0], VR.shape[0]))
    SG_compressed = np.zeros((UG.shape[0], VG.shape[0]))
    SB_compressed = np.zeros((UB.shape[0], VB.shape[0]))

    for k in kArr:
        try:
            # truncates the S matrices into kxk matrices and assigns them to other kxk sized matrices S*_compressed
            SR_compressed[:k, :k] = SR[:k, :k]
            SG_compressed[:k, :k] = SG[:k, :k]
            SB_compressed[:k, :k] = SB[:k, :k]

            # multiplies the 3 SVD matrices back into the R, G, and B matrices
            compressedR = UR @ SR_compressed @ VR
            compressedG = UG @ SG_compressed @ VG
            compressedB = UB @ SB_compressed @ VB

            compressedImg = np.stack((compressedR, compressedB, compressedG), axis=-1) # stacks the 3 color channels back into a single image matrix

            save_compressed_image(compressedImg, k)

            imgSizeCmp = get_image_size(f'comp{k}.jpg')
            cmpRatio = imgSizeOG / imgSizeCmp
            print(f'[k={k}] File Size: {imgSizeCmp} kilobytes | Compression Ratio: {round(cmpRatio, 4)}')

        except ValueError:
            print(f'k larger than singular values matrix (k={k})')


try:
    with Image.open(imgFile) as img:
        isRunning = True

        while isRunning:
            colorPref = input('Color compression (c) or greyscale (g)?: ')

            if colorPref == 'g':
                compress_image_greyscale(img)
                isRunning = False
            elif colorPref == 'c':
                compress_image_color(img)
                isRunning = False
            else:
                print('Invalid option')

except IOError as e:
    print(f'Error raised: {e}')
