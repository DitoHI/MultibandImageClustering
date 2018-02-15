from django.shortcuts import render
from PIL import Image
import numpy as np
import copy as cp
import random
import math
import time
from .forms import selectClust
from collections import Counter
from django.http import HttpResponse
from django.conf import settings

img_dir = "/static/spectral/"
result_dir = img_dir + "result/"
pic_dir = "/static/spectral_img/"
w, h = 50, 50
sumPic = 6
sumClust = 3
dimCol = 3
iteration = 10

colorValues = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]
]

# Create your views here.
def home(request):
    start = time.time()
    time.clock()

    #form
    global sumClust
    form = selectClust(request.POST or None)
    sumClust = 3
    title = "Project to show an implementation of clustering on spectral"
    message = None
    result_url = None
    if form.is_valid():
        cluster = form.cleaned_data['clust']
        sumClust = cluster
        title = None
        message = "Search Time wasted around"

    # Function ONE // return value: directory of image and the pixel
    img, pic, pixel = getPixel()

    if message != None :
        # Function TWO // return value: get the value of fst
        fst = makefst(pixel)
        # Function THREE // return value: get cluster
        cluster = kmeans_ga(fst)
        # Function FOUR // return value: get label
        color, label = labeling(fst, cluster)

        # convert the color to image based on height and width
        color = np.reshape(color, (w, h, dimCol))
        data = np.array(color, dtype=np.uint8)
        image = Image.fromarray(data, 'RGB')
        result_url = result_dir + "result.png"
        temp_dir = settings.STATIC_ONLY + result_url
        image.save(temp_dir)

    elapsed = time.time() - start
    elapsed = round(elapsed, 2)
    time.sleep(1)
    context = {
        'form' : form,
        'url': pic,
        'result' : result_url,
        'cluster' : sumClust,
        'elapsed' : elapsed,
        'title' : title,
        'message' : message,
    }
    template = 'multiband_home.html'
    return render(request, template, context)

#Function ONE
def getPixel():
    pixelPic = np.zeros(shape=(sumPic, w*h))
    img_url = [0 for x in range(sumPic)]
    pic_url = [0 for x in range(sumPic)]
    for i in range(0,sumPic):
        index = cp.deepcopy(i) + 1
        if (index == sumPic) :
            index += 1
        #Get the directory of Landshape (six image)
        img_url[i] = img_dir + "gb" + str(index) + ".gif"
        pic_url[i] = pic_dir + "gb" + str(index) + ".jpg"
        url = settings.STATIC_ONLY + img_url[i]
        img = Image.open(url)
        #Get the pixel from PIC
        img_data = np.array(img, dtype=np.float64)
        pixelPic[i] = cp.deepcopy(img_data.ravel())

    return img_url, pic_url, pixelPic

#Function TWO
def makefst(arrpixel):
    d = cp.deepcopy(sumPic) #define the dimension of space
    newfst = np.zeros(shape=((w*h), d)) #initialize feature space transformation
    for i in range(0, w*h):
        for j in range(0, d):
            val = cp.deepcopy(arrpixel[j][i])
            newfst[i][j] = cp.deepcopy(val)

    return newfst

#Function Three
def kmeans_ga(fst):
    w_fst, h_fst = tuple(fst.shape)
    sumInd = 8

    new_fst = np.zeros(shape=(sumPic, w*h))
    for i in range(0, sumPic):
        for j in range(0, len(fst)):
            new_fst[i][j] = cp.deepcopy(fst[j][i])

    arr_max = new_fst.max(axis=1) #min value
    arr_min = new_fst.min(axis=1) #max value

    newInd = np.zeros(shape=(sumInd, sumClust, h_fst)) #initialize individu
    ind = getInitializeInd(newInd, arr_min, arr_max) #get random individu => array(8,3,6)

    #do the iteration
    for i in range(0, iteration):
        fitness = fitnessCalc(fst, ind, sumInd)  # get fitness value from each individu
        ind_roulette, fit_roulette = roulette(ind, fitness, sumInd)  # roulette the fitness
        ind_crossover = crossover(ind_roulette, sumInd)  # crossover the individual from result of roulette
        ind_mutation = mutation(ind_crossover, sumInd)  # mutation of the indvidual from result of crosspover
        fitness_mutation = fitnessCalc(fst, ind_mutation, sumInd)  # get fitness value from individu #2
        ind_result = elistism(ind, fitness, ind_mutation, fitness_mutation, sumInd) # get the individual with highest fitness
        ind = cp.deepcopy(ind_result)

    return ind[0]

def getInitializeInd(ind, arr_min, arr_max):
    #ind = array(8, 3, 6)
    for i, item1 in enumerate(ind):
        for j, item2 in enumerate(item1):
            for k, item3 in enumerate(item2):
                x = random.uniform(arr_min[k], arr_max[k])
                ind[i][j][k] = cp.deepcopy(x)

    return ind

def fitnessCalc(fst, ind, sum):
    temp_ind = cp.deepcopy(ind)
    dist, tot, sumTot = 0, 0, 0
    fitness = np.zeros(shape=sum)
    fitpercen = np.zeros(shape=sum)
    # temp_ind = np.reshape(temp_ind, (sum, dimens))

    # fst = array(1024, 6)
    # temp_ind = array(8, 3, 6)
    for i in range(0, sum):
        for j in range(0, len(fst)):
            for k in range(0, len(temp_ind[i])):
                temp = 0
                for l in range(0, len(temp_ind[i][k])):
                    temp = temp + math.pow((temp_ind[i][k][l] - fst[j][l]), 2)
                temp_sqrt = math.sqrt(temp)
                if dist == 0:
                    dist = temp_sqrt
                elif dist != 0 :
                    if dist > temp_sqrt :
                        dist = temp_sqrt
            tot += dist
            dist = 0
        fitness[i] = 1 / tot
        tot = 0
        sumTot += fitness[i]

    for i in range(0, sum):
        fitpercen[i] = (fitness[i] / sumTot) * 100

    return fitpercen

def roulette(ind, fitness, sum):
    newInd = np.zeros(shape=(sum, sumClust, sumPic))
    newfitness = np.zeros(shape=(sum))
    for i in range(0, sum):
        count = random.uniform(1, 100)
        temp1 = fitness[0]
        temp2 = temp1
        if count <= temp1 :
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[0][j][k])
            newfitness[i] = fitness[0]
        temp2 += fitness[1]
        if temp1 < count <= temp2 :
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[1][j][k])
            newfitness[i] = fitness[1]
        temp1 = temp2
        temp2 += fitness[2]
        if temp1 < count <= temp2:
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[2][j][k])
            newfitness[i] = fitness[2]
        temp1 = temp2
        temp2 += fitness[3]
        if temp1 < count <= temp2:
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[3][j][k])
            newfitness[i] = fitness[3]
        temp1 = temp2
        temp2 += fitness[4]
        if temp1 < count <= temp2:
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[4][j][k])
            newfitness[i] = fitness[4]
        temp1 = temp2
        temp2 += fitness[5]
        if temp1 < count <= temp2:
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[5][j][k])
            newfitness[i] = fitness[5]
        temp1 = temp2
        temp2 += fitness[6]
        if temp1 < count <= temp2:
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[6][j][k])
            newfitness[i] = fitness[6]
        temp1 = temp2
        temp2 += fitness[7]
        if temp1 < count <= temp2:
            for j in range(0, len(ind[i])):
                for k in range(0, len(ind[i][j])):
                    newInd[i][j][k] = cp.deepcopy(ind[7][j][k])
            newfitness[i] = fitness[7]

    return newInd, newfitness

def crossover(ind, sum):
    prob = 0.9

    dimens = sumClust*sumPic
    temp_ind = cp.deepcopy(ind)
    temp_ind = np.reshape(temp_ind, (sum, dimens))

    for i in range(0, sum, 2):
        probCross = random.uniform(0, 100)
        probCross = probCross / 100
        poin = random.randint(0, (dimens - 1))
        s = dimens - poin
        length = random.randint(1, s)
        if (probCross < prob):
            for j in range(poin, (poin + length)):
                temp = temp_ind[i][j]
                temp_ind[i][j] = temp_ind[i+1][j]
                temp_ind[i+1][j] = temp

    temp_ind = np.reshape(temp_ind, (sum, sumClust, sumPic))

    return temp_ind

def mutation(ind, sum):
    prob = 0.3
    n = 10

    dimens = sumClust * sumPic
    temp_ind = cp.deepcopy(ind)
    temp_ind = np.reshape(temp_ind, (sum, dimens))

    for i in range(0, sum):
        mutProb = random.uniform(0, 100)
        mutProb = mutProb / 100
        if mutProb < prob:
            index = random.randint(0, (dimens-1))
            tipeProb = random.randint(1, 100)
            if tipeProb < 50:
                n *= -1

            temp_ind[i][index] += n
            n = 10

    temp_ind = np.reshape(temp_ind, (sum, sumClust, sumPic))
    return temp_ind

def elistism(ind_pure, fitness_pure, ind_mutation, fitness_mutation, sum):
    dimens = sumClust * sumPic

    temp_ind_pure = cp.deepcopy(ind_pure)
    temp_ind_pure = np.reshape(temp_ind_pure, (sum, dimens))

    temp_ind_mutation = cp.deepcopy(ind_mutation)
    temp_ind_mutation = np.reshape(temp_ind_mutation, (sum, dimens))

    temp_ind_result = np.append(temp_ind_pure, temp_ind_mutation, axis=0).tolist()
    temp_fitness_result = np.append(fitness_pure, fitness_mutation, axis=0).tolist()

    #sorting individu based on fitness
    ind_result = [temp_ind_result for _, temp_ind_result in sorted(zip(temp_fitness_result, temp_ind_result))]
    ind_result = ind_result[0:8]
    ind_result = np.array(ind_result, dtype=np.float64)

    ind_result = np.reshape(ind_result, (sum, sumClust, sumPic))

    return ind_result

#Function Four
def labeling(fst, cluster):
    temp_fst = cp.deepcopy(fst)
    label = np.zeros(shape=(len(temp_fst)))
    for i in range(0, len(temp_fst)):
        dist = 0
        for j in range(0, len(cluster)):
            temp = 0
            for k in range(0, len(cluster[j])):
                temp += math.pow((cluster[j][k] - temp_fst[i][k]), 2)
            temp_sqrt = math.sqrt(temp)
            if dist == 0:
                dist = temp_sqrt
                label[i] = cp.deepcopy(j)
            elif dist != 0:
                if dist > temp_sqrt:
                    dist = temp_sqrt
                    label[i] = cp.deepcopy(j)

    label = label.astype(int)
    pixelCol = np.zeros(shape=(len(temp_fst), dimCol))
    for i in range(0, len(label)):
        for j in range(0, dimCol):
            pixelCol[i][j] = colorValues[label[i]][j]

    return pixelCol, label







