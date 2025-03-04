import os
import matplotlib.pyplot as plt

rpath = 'compressed-images/results.txt'
spath = 'compressed-images/k-size.jpg'

kVal = []
fsize = []
cmpR = []
psnr = []

# k-size, k-cmpR, k-psnr

def create_graph() -> None:
    if os.path.exists(rpath):
        with open(rpath) as file:
            data = file.readlines()

            for line in data:
                datan = line
                datan = list(map(str, datan.split(' ')))
                datan.remove('\n')
                datan = list(map(float, datan))

                kVal.append(datan[0])
                fsize.append(datan[1])
                cmpR.append(datan[2])
                psnr.append(datan[3])

        plt.scatter(kVal, fsize)
        plt.savefig(spath)

    else:
        print('File does not exist')
