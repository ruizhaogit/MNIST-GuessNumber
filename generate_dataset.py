
# coding: utf-8


import numpy as np
from mnist_util import read
from num2words import num2words
import operator as op
import json
import scipy.misc
import os


# possible properties
colors = ['blue', 'red', 'green', 'violet', 'brown']
bgcolors = ['white', 'cyan', 'salmon', 'yellow', 'silver']
styles = ['flat', 'stroke']
properties = ['number', 'color', 'bgcolor', 'style']



def generateGridImg(size=3):
    while True:
        img = []
        for i in range(size):
            img.append([])
            for j in range(size):
                cell = {}
                cell['number'] = np.random.randint(10)
                cell['color'] = colors[np.random.randint(len(colors))]
                cell['bgcolor'] = bgcolors[np.random.randint(len(bgcolors))]
                cell['style'] = styles[np.random.randint(len(styles))]
                img[i].append(cell)
        img_flat = [x for sublist in img for x in sublist]
        ## check if all numbers are unique in the image
        if len(img_flat) == len([dict(s) for s in set(frozenset(d.items()) for d in img_flat)]):
            break

    return img



def printGridImg(img):
    for i in range(len(img)):
        for k in img[0][0]:
            for j in range(len(img[i])):
                print( k, img[i][j][k], '\t', end='') 
            print()
        print()



def initTargetMap(size=3):
    targetMap = []
    for i in range(size):
        targetMap.append([])
        for j in range(size):
            targetMap[i].append(True)
    return targetMap



def getChecklist(gridImg, targetMap):
    checklist = {}
    for k in properties:
        checklist[k] = set()
        for i in range(len(gridImg)):
            for j in range(len(gridImg[i])):
                if targetMap[i][j]:
                    checklist[k].add(gridImg[i][j][k])
    
    return checklist



def updateChecklist(checklist, gridImg, targetMap, reinit = False):
    for k in checklist:
        if not reinit and len(checklist[k]) == 0:
            continue
        
        checklist[k] = set()
        for i in range(len(gridImg)):
            for j in range(len(gridImg[i])):
                if targetMap[i][j]:
                    checklist[k].add(gridImg[i][j][k])



def noChecklist(checklist):
    for k in checklist:
        if len(checklist[k]) != 0:
            return False
    return True



def getTargets(gridImg, targetMap, prop):
    count = 0
    targetIndices = []
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            if targetMap[i][j]:
                targetIndices.append((i, j))
    return targetIndices



def countTargets(gridImg, targetMap, prop, val):
    count = 0
    targetIndices = []
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            if targetMap[i][j] and gridImg[i][j][prop] == val:
                count += 1
                targetIndices.append((i, j))
    return count, targetIndices



def moveTarget(gridImg, targetMap, index, direction):
    new_index = list(index)
    
    if direction == 0:
        new_index[0] -= 1
    elif direction == 1:
        new_index[0] += 1
    elif direction == 2:
        new_index[1] -= 1
    elif direction == 3:
        new_index[1] += 1
        
    targetMap[index[0]][index[1]] = False
    targetMap[new_index[0]][new_index[1]] = True
    
    return tuple(new_index)



def selectSubTargetMap(gridImg, targetMap, prop, val, reverse=False):
    count = 0
    targetIndices = []
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            if targetMap[i][j] and gridImg[i][j][prop] != val and not reverse:
                targetMap[i][j] = False
            elif targetMap[i][j] and gridImg[i][j][prop] == val and reverse:
                targetMap[i][j] = False


def reinitTargetMap(targetMap):
    for i in range(len(targetMap)):
        for j in range(len(targetMap[i])):
            targetMap[i][j] = True


# # QA instance generation

def initTarNum(gridImg):
    i = np.random.randint(len(gridImg))
    j = np.random.randint(len(gridImg))
    tarNumIndice = (i, j)
    return tarNumIndice

def getNumMap(targetMap):
    count = 0
    for i in range(len(targetMap)):
        for j in range(len(targetMap[i])):
            if targetMap[i][j] == True:
                count = count + 1
    return count


def genQA_a(gridImg, targetMap, checklist, tarNum, subset=False):
    while True:
        prop_class = np.random.multinomial(1, [0.4, 0.25, 0.25, 0.1]).argmax()
        if len(checklist[properties[prop_class]]) != 0:
            break

    possible_no = np.random.multinomial(1, [0.98, 0.02]).argmax()
    template = 'Is it %s ?'
            
    
    if prop_class == 0: # number
        if possible_no == 1:
            num = np.random.randint(10)
        else:
            num = list(checklist['number'])[np.random.randint(len(checklist['number']))]
        template = template % '%s in the image'
        question = template % ("%d" % num)
        val = num
        
    elif prop_class == 1: # color
        if possible_no == 1:
            color = colors[np.random.randint(len(colors))]
        else:
            color = list(checklist['color'])[np.random.randint(len(checklist['color']))]
        question = template % ('a digit in %s' % color)
        val = color
        
    elif prop_class == 2: # bgcolor
        if possible_no == 1:
            bgcolor = bgcolors[np.random.randint(len(bgcolors))]
        else:
            bgcolor = list(checklist['bgcolor'])[np.random.randint(len(checklist['bgcolor']))]
        question = template % ('in a %s background' % bgcolor)
        val = bgcolor
        
    elif prop_class == 3: # style
        if possible_no == 1:
            style = styles[np.random.randint(len(styles))]
        else:
            style = list(checklist['style'])[np.random.randint(len(checklist['style']))]
        if style == 'flat':
            question = template % 'a flat style digit'
        elif style == 'stroke':
            question = template % 'a stroke style digit'
        val = style
    count, targetIndices = countTargets(gridImg, targetMap, properties[prop_class], val)
    if not (tarNum in targetIndices):
        answer = 'no'
        selectSubTargetMap(gridImg, targetMap, properties[prop_class], val, True)
        updateChecklist(checklist, gridImg, targetMap)
    else:
        answer = 'yes'
        checklist[properties[prop_class]] = set()
        selectSubTargetMap(gridImg, targetMap, properties[prop_class], val)
        updateChecklist(checklist, gridImg, targetMap)
    
    return question, answer, ('Qa', prop_class, val), tuple(targetIndices)


# # QA sequence generation

def generateSeqQA(gridImg, tarNum, maxlen = 10):
    targetMap = initTargetMap(len(gridImg))
    checklist = getChecklist(gridImg, targetMap)
    
    state = 1
    QASequence = []
    prev_state = None
    prev_index = None
    while True:            
        QAInfo = {}
        QAInfo['question'], QAInfo['answer'], _, _ = genQA_a(gridImg, targetMap, checklist, tarNum)
        
        QASequence.append(QAInfo)
        if getNumMap(targetMap) == 1:
            break
    if len(QASequence) > data['maxlen']:
        data['maxlen'] = len(QASequence)
    return QASequence


# # Image realization


import matplotlib.pyplot as plt
from scipy.ndimage.morphology import grey_dilation



def getNumPools():
    pools = {}
    for dataset in ['training', 'testing']:
        mnist = read(path='data/mnist')
        mnist = sorted(mnist, key = op.itemgetter(0))
        mnist = [(x[0], x[1].astype('float32')/255) for x in mnist]
        
        numPools= []
        for i in range(9):
            count = 0
            for j in range(len(mnist)):
                if mnist[j][0] != i:
                    break
                count+=1
            numPools.append(mnist[:count])
            mnist = mnist[count:]
        numPools.append(mnist)
        
        pools[dataset] = numPools
    return pools



numPools = getNumPools()
colorMap = {}
colorMap['blue'] = [49, 89, 191]
colorMap['red'] = [186, 29, 18]
colorMap['green'] = [62, 140, 33]
colorMap['violet'] = [130, 58, 156]
colorMap['brown'] = [119, 57, 19]
colorMap['white'] = [255, 255, 255]
colorMap['cyan'] = [71, 255, 253]
colorMap['salmon'] = [255, 173, 148]
colorMap['yellow'] = [252, 251, 100]
colorMap['silver'] = [204, 204, 204]
for k in colorMap:
    colorMap[k] = np.array(colorMap[k], dtype='float32').reshape((1, 1, 3)) / 255



def realizeSingleNumber(info, size = 28, dataset='training'):
    palette = np.ones((size, size, 3), dtype='float32') * colorMap[info['bgcolor']]
    
    num_sample_idx = np.random.randint(len(numPools[dataset][info['number']]))
    num_sample = numPools[dataset][info['number']][num_sample_idx][1]
    
    if info['style'] == 'stroke':
        mask = grey_dilation(num_sample, (3,3)).reshape((size, size, 1))
        palette = palette * (1-mask)
    
    mask = num_sample.reshape((size, size, 1))
    palette = palette * (1-mask) + (mask * colorMap[info['color']]) * mask
    
    return palette



def realizeGrid(gridImg, size=28, dataset='training'):
    img = np.zeros((size*len(gridImg), size*len(gridImg), 3))
    for i in range(len(gridImg)):
        for j in range(len(gridImg[i])):
            img[i*size:(i+1)*size, j*size:(j+1)*size, :] = realizeSingleNumber(gridImg[i][j], size=size, dataset=dataset)
    return img


# # Dataset Generation

# parameters
data = {}
data['seed'] = 123
data['maxlen'] = 0
data['gridSize'] = 3
data['trainSize'] = 30000
data['validSize'] = 10000
data['testSize'] = 10000
data['QAperImg'] = 1
np.random.seed(data['seed'])
imgPath = 'data/mnist_guess_number/imgs'
jsonPath = 'data/mnist_guess_number'



if not os.path.exists(jsonPath):
    os.makedirs(jsonPath)

for split in ['train', 'valid', 'test']:
    splitPath = os.path.join(imgPath, split)
    if not os.path.exists(splitPath):
        os.makedirs(splitPath)
    examples = []
    gridImgs = []
    for i in range(data[split+'Size']):
        # print(i)
        gridImg = generateGridImg(data['gridSize'])
        img = realizeGrid(gridImg, dataset=(split == 'test' and 'testing' or 'training'))
        gridImgs.append(gridImg)
        scipy.misc.imsave(os.path.join(splitPath, '%05d.jpg'%i), img)
        for j in range(data['QAperImg']):
            tarNum = initTarNum(gridImg)
            qa = generateSeqQA(gridImg, tarNum)
            examples.append({'img':i, 'qa':qa, 'target':tarNum, 'gridImg':gridImg})

    with open(os.path.join(jsonPath, split+'.json'), 'w') as f:
        for example in examples:
            json.dump(example, f)
            f.write('\n')


img = generateGridImg(data['gridSize'])
targetMap = initTargetMap(data['gridSize'])
checklist = getChecklist(img, targetMap)
printGridImg(img)
plt.imsave('test.jpg', realizeGrid(img))
