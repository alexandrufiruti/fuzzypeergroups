# Proiect PAIC
# Implementare filtru Fuzzy Peer Group Averaging (FPGA)

# Includes
import numpy as np 
from skimage import io,color 
import skimage
import matplotlib.pyplot as plt
import operator
import math

# Globals
Fsigma = 100
Ft = 0.15
n = 3 # nxn dimensiunea ferestrei de filtrare
nSq = pow(n, 2)

affectedPixels = []
detectedAffectedPixels = []

def marginal_median_filter(I):
    # Filtru median marginal
    newI = I.copy()
    ch = I.shape[2]
    for i in range(1, I.shape[0]-1):
        for j in range(1, I.shape[1]-1):
            red = np.sort(I[i-1:i+2,j-1:j+2,0], axis=None)[4]
            green = np.sort(I[i-1:i+2,j-1:j+2,1], axis=None)[4]
            blue = np.sort(I[i-1:i+2,j-1:j+2,2], axis=None)[4]
            newI[i,j,:] = np.array([red,green,blue])
    return newI

def marginal_arithmetic_mean_filter(I):
    # Filtru de medie aritmetica marginal de dimensiune 3x3
    newI = I.copy()
    ch = I.shape[2]
    for i in range(1, I.shape[0]-1):
        for j in range(1, I.shape[1]-1):
            redMean = int(np.mean(I[i-1:i+2,j-1:j+2,0]))
            greenMean = int(np.mean(I[i-1:i+2,j-1:j+2,1]))
            blueMean = int(np.mean(I[i-1:i+2,j-1:j+2,2]))
            newI[i,j,:] = np.array([redMean,greenMean,blueMean])
    return newI

def add_mixed_noise(I):
    mean = 0
    stdDev = 10
    gaussianNoise = np.random.normal(mean,stdDev,[I.shape[0],I.shape[1],3])
    I = I + gaussianNoise
    I = np.clip(I, 0, 255)
    I = I.astype('uint8')
    pImpNoise = 0.1 # procent de pixeli afectati de zgomot impulsiv per canal de culoare
    N = int(pImpNoise*(I.shape[0]*I.shape[1])/3 ) # numar de pixeli afectati de zgomot impulsiv
    for channel in range(3):
        for j in range(N):
            linePos=np.random.randint(0,I.shape[0], [1,1])
            linePos=int(linePos)
            colPos=np.random.randint(0,I.shape[1], [1,1])
            colPos=int(colPos)
            I[linePos,colPos,channel]=(int(np.random.randint(0, 2, [1,1])))*255 # random black or white
            affectedPixels.append((linePos,colPos))
    return I

# functie de similaritate fuzzy
def fuzzy_similarity_function(Fi, Fj):
    return np.exp((-1)*(np.linalg.norm(Fi-Fj, ord=2)/Fsigma))
    
#algoritmul de filtrare (reducere de zgomot mixt - impulsiv si gaussian)
def fuzzy_peer_groups_algorithm(I, x, y):
    # window = I[i-borderSize:i+borderSize+1, j-borderSize:j+borderSize+1, :]
    F0 = I[x,y,:]
    # un dictionar in care se retin coordonatele pixelului (sub forma de cheie) 
    # si functia de similaritate asociata (sub forma de valoare)
    similarityDict = dict()
    R = G = B = [] # pentru determinarea medianului marginal
    
    for i in range(x-borderSize, x+borderSize+1):
        for j in range(y-borderSize, y+borderSize+1):
            Fj = I[i,j,:]
            R.append(Fj[0])
            G.append(Fj[1])
            B.append(Fj[2])
            similarityDict[(i,j)] = fuzzy_similarity_function(F0, Fj)

    # Se sorteaza descrescator pixelii in functie de similaritate
    sorted_similarityDict = dict(sorted(similarityDict.items(), key=operator.itemgetter(1),reverse=True))
    # C este functia ce arata gradul de apartenenta a pixelului la grupul de similaritate (fct descrescatoare)
    C = list(sorted_similarityDict.values())
    # A este functia de asimilare de similaritate a pixelului (fct crescatoare)
    A = [sum(C[:i+1]) for i in range(len(C))]
    # L este o functie cuadratica L(F(i)) = u(A(F(i)))
    L = [(-(1/(pow(nSq-1, 2)))*(A[i]-1)*(A[i]-2*nSq+1)) for i in range(len(A))]
    # C_FR1 este certitudinea (Fuzzy Rule 1)
    C_FR1 = [C[m]*L[m] for m in range(1, nSq)]
    # m_optim este m pentru care C_FR1 are valoare maxima
    m_optim = C_FR1.index(max(C_FR1))+1
    # C_FR2 este certitudinea pixelului central (Fuzzy Rule 2)
    C_FR2 = C_FR1[m_optim-1]
    if (C_FR2 < Ft):
        # Pixelul F0 este afectat de zgomot. Trebuie inlocuit prin VMF
        detectedAffectedPixels.append((x, y))
        # Se implementeaza filtrarea vectoriala prin abordare "distanta"
        R.sort()
        G.sort()
        B.sort()
        neighbourPixels = dict()
        medianPixel = (R[math.ceil(nSq/2)],G[math.ceil(nSq/2)],B[math.ceil(nSq/2)])
        for i in range(x-borderSize, x+borderSize+1):
            for j in range(y-borderSize, y+borderSize+1):
                neighbourPixels[(i,j)] = np.linalg.norm(I[i,j,:]-medianPixel, ord=2)
        sorted_neighbourPixels = dict(sorted(neighbourPixels.items(), key=operator.itemgetter(1)))
        xy = list(sorted_neighbourPixels.keys())[0]
        pixelWithoutNoise = I[xy[0],xy[1],:]
    else:
        pixelWithoutNoise = F0
    
    pixelsValuesInFuzzyPeerGroup = [I[i] for i in list(sorted_similarityDict.keys())[:m_optim]]
    weightsInFuzzyPeerGroup = list(sorted_similarityDict.values())[:m_optim]
    numerator = pixelWithoutNoise
    denominator = 1
    for i in range(1, m_optim):
        numerator = numerator + weightsInFuzzyPeerGroup[i]*pixelsValuesInFuzzyPeerGroup[i]
        denominator = denominator + weightsInFuzzyPeerGroup[i]
    pixelWithoutNoise = numerator/denominator
    pixelWithoutNoise = [int(x) for x in pixelWithoutNoise]
    pixelWithoutNoise = np.clip(pixelWithoutNoise, a_min=0, a_max=255)
    return pixelWithoutNoise

img = io.imread('lena.png')
imgWithNoise = img.copy()
imgWithNoise = add_mixed_noise(imgWithNoise)
imgWithoutNoise = imgWithNoise.copy()
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.title("Imagine originala")
plt.figure(figsize=(10,10))
plt.imshow(imgWithNoise)
plt.title("Imagine afectata de zgomot mixt")

plt.figure(figsize=(10,10))
I1 = marginal_median_filter(imgWithNoise)
plt.imshow(I1)
plt.title("Imagine dupa filtru median marginal")

plt.figure(figsize=(10,10))
I2 = marginal_arithmetic_mean_filter(I1) 
plt.imshow(marginal_median_filter(imgWithNoise))
plt.title("Imagine dupa filtru de medie aritmetica marginal")


# se parcurge imaginea pixel cu pixel 
# (excluzand bordura de (n-1)/2 pixeli, nxn este dimensiunea ferestrei de filtrare)

borderSize = int((n-1)/2)
for i in range(borderSize, imgWithNoise.shape[0]-borderSize):
    for j in range(borderSize, imgWithNoise.shape[1]-borderSize):
        # F0 = imgWithNoise[i,j,:] (pixelul central)
        imgWithoutNoise[i,j,:] = fuzzy_peer_groups_algorithm(imgWithNoise, i, j)
         
plt.figure(figsize=(10,10))
plt.imshow(imgWithoutNoise) 
plt.title("Imagine dupa filtrarea cu algoritmul specificat")

print("")
print("PSNR pentru filtrarea clasica cu filtru median marginal: {}".format(skimage.measure.compare_psnr(img,I1)))
print("PSNR pentru filtrarea clasica cu filtru de medie aritmetica marginal: {}".format(skimage.measure.compare_psnr(img,I2)))
print("PSNR pentru filtrarea cu algoritmul specificat: {}".format(skimage.measure.compare_psnr(img,imgWithoutNoise)))
print("")
print("MSE pentru filtrarea clasica cu filtru median marginal: {}".format(skimage.measure.compare_mse(img,I1)))
print("MSE pentru filtrarea clasica cu filtru de medie aritmetica marginal: {}".format(skimage.measure.compare_mse(img,I2)))
print("MSE pentru filtrarea cu algoritmul specificat: {}".format(skimage.measure.compare_mse(img,imgWithoutNoise)))
