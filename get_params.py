from math import sqrt
from scipy import interpolate
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numbers
from ADRpy import atmospheres as at

# EM
# https://hobbyking.com/es_es/propdrive-v2-5050-580kv-brushless-outrunner-motor.html?queryID=c0c7923d37f16fea4d4032ef5475e7f9&objectID=59101&indexName=hbk_live_magento_es_es_products

# List of LiPos
# https://hobbyking.com/es_es/graphene-1300mah-4s-75c.html
# https://hobbyking.com/es_es/turnigy-2200mah-3s-30c-lipo-pack.html?queryID=6edbaf7d955a89c783059f6bf2aa2ac2&objectID=18298&indexName=hbk_live_magento_es_es_products
# https://hobbyking.com/es_es/zippy-compact-2200mah-4s-25c-lipo-pack.html?queryID=972631537352fbd4903c784d6c5419b9&objectID=24697&indexName=hbk_live_magento_es_es_products
# https://hobbyking.com/es_es/turnigy-high-capacity-12000mah-4s-12c-multi-rotor-lipo-pack-xt90.html?queryID=655f4df519571a45ebce53620a2ccea4&objectID=79996&indexName=hbk_live_magento_es_es_products
# https://hobbyking.com/es_es/turnigy-high-capacity-battery-20000mah-4s-12c-drone-lipo-pack-xt90.html?queryID=0fa6a706a6d21abd038e864fa056bebf&objectID=78415&indexName=hbk_live_magento_es_es_products


def Density(h):
    """
    Air density at a given altitude based on ideal gas law

    Parameters
    ----------
    h : int or float
        Altitude.

    Raises
    ------
    ValueError
        When wrong arg type is passed.

    Returns
    -------
    rho : float
        Density at altitude.

    """
    
    if not isinstance(h, numbers.Number):
        raise ValueError('Wrong arg type passed to Density() for h. Must be'
                         + ' int or float')

    return at.Atmosphere().airdens_kgpm3(h)


# <codecell> get_data function declarations
def ArrayMax2d(array):
    """ Returns the maximum value and the indeces of the 2d array that was
        passed. """
    if len(np.shape(array)) != 2:
        raise ValueError('ArrayMax2d was passed an array that is not',
                         '2-dimensional')
    
    maxVal = 0
    for index, i in enumerate(array[0, :]):
        for jndex, j in enumerate(array[:, index]):
            if j > maxVal:
                maxVal = j
                row = index
                col = jndex
    
    return maxVal, row, col


def ReadArray(f, dimensions):
    global nRows, nCols, array
    nRows = ''
    nCols = ''
    rows = True
    # determine the dimensions of the array that will be read
    for letter in dimensions:
        if letter == 'x':
            rows = False
        # attach to number of rows
        if rows and letter != 'x':
            nRows += letter
            
        if not rows and letter != 'x':
            nCols += letter
    
    nRows = int(nRows)
    nCols = int(nCols)
    
    # create array to fill in with correct values
    array = np.zeros((nRows, nCols))
    
    # read in array
    for i in range(nRows):
        line = f.readline()
        line = [float(j) for j in line.split(',')]
        array[i, :] = line
        
    if nRows == 1:
        array = array[0]

    return array


def FilterString(f, string):
    global tempName, tempValue
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'x']
    tempName = ''
    tempValue = ''
    varName = False
    value = False
    for index, i in enumerate(string):
        # check if variable name ends
        if i == '}':
            varName = False
        # append to variable name
        if varName:
            tempName += i
            
        # append to variable value
        elif i in nums and value:
            tempValue += i
        
        # check if variable name begins
        if i == '{':
            varName = True
        # check if variable value begins
        elif i == ':':
            value = True
    
    if tempName == '' or tempName == '':
        raise ValueError(string + ' does not contain numbers')
        
    elif 'x' in tempValue:
        tempValue = ReadArray(f, tempValue)
        
    else:
        tempValue = float(tempValue)
            
    return str(tempName), tempValue


def EmptyLine(line):
    return line == '\n'


def ReadNextLine(f, line):
    while EmptyLine(line) or '---' in line:
        line = f.readline()
        
    return FilterString(f, line)


# <codecell> get parameter data
aircraft_dict = {}
plant_dict = {}


def GetParams(fileName='aircraft_params.txt'):
    global aircraft_dict
    # open the file
    f = open(fileName, 'r')

    
    # go through the file to get all the variables
    for line in f:
        varName, varValue = ReadNextLine(f, line)
        # print(varName, '=', varValue)
        
        # check which dictionary to store values in
        if varName == 'dict' and varValue == 0:
            reading = 'airframe'
        if varName == 'dict' and varValue == 1:
            reading = 'powerplant'
        
        # store values in correct dictionary
        if reading == 'airframe':
            aircraft_dict.update({varName: varValue})
        elif reading == 'powerplant':
            plant_dict.update({varName: varValue})
        
        # globals()[varName] = varValue

    # <codecell> process parameter data
    # calculate the wingspan from given fata
    span = sqrt(aircraft_dict['AR'] * aircraft_dict['S'])
    aircraft_dict.update({'span': span})

    # maximum value of the efficiency array and the corresponding
    # torque and shaft speed
    maxICEeff, row, col = ArrayMax2d(plant_dict['effMap'])
    
    # Torque and shaft speed which will produce maximum efficiency for the ICE
    idealICETorque = plant_dict['ICEMapTorque'][col]
    idealICErps = plant_dict['ICEMaprps'][row]
    
    
    # shaft power produced at this ideal torque and rps
    idealICEPower = 2 * math.pi * idealICErps * idealICETorque
    
    # max prop efficiency and matching J
    maxPropEff = np.amax(plant_dict['propEffData'])
    optJ = plant_dict['JIndex'][int(np.where(plant_dict['propEffData'] == maxPropEff)[0])]
    
    plant_dict.update({'maxICEeff': maxICEeff,
                       'idealICETorque': idealICETorque,
                       'idealICErps': idealICErps,
                       'idealICEPower': idealICEPower,
                       'maxPropEff': maxPropEff,
                       'optJ': optJ})
                
    
    PropCt = interp1d(plant_dict['JIndex'], plant_dict['propCtData'],
                      fill_value='extrapolate')
    
    PropCq = interp1d(plant_dict['JIndex'], plant_dict['propCpData'] / (2 * math.pi),
                      fill_value='extrapolate')
    
    PropCp = interp1d(plant_dict['JIndex'], plant_dict['propCpData'],
                      fill_value='extrapolate')

    # joules capacity = V[V] * capacity[mAh] * 3.6[A*s/(mA*h)]
    battCapList = plant_dict['battVList'] * plant_dict['battCapList'] * 3.6

    # specific energy [joules/kg] = capacity[joules] / batt mass[kg]
    se = battCapList / plant_dict['battMassList']

    plant_dict.update({'battSEList': se,
                       'battCapList': battCapList,
                       'CtFun': PropCt,
                       'CqFun': PropCq,
                       'CpFun': PropCp
                       })

    return aircraft_dict, plant_dict


# <codecell> Thrust and power maps
def MakeMaps():
    maxMaprps = max((plant_dict['maxEMrps'], plant_dict['ICEMaprps'][-1]))
    rpsArray = np.linspace(0, maxMaprps, 20)
    # JArray is limited to the values that come with the propeller data
    JArray = np.linspace(0, plant_dict['JIndex'][0], 20)
    hArray = np.linspace(0, 4000, 20)
    
    TMap = np.zeros((20, 20, 20))
    
    propCt = interpolate.interp1d(plant_dict['JIndex'], plant_dict['propCtData'],
                                  fill_value='extrapolate')
    
    # fill TMap with rpsArray as the row index, JArray as the column index and h
    # as the depth axis
    for hndex, h in enumerate(hArray):
        for index, i in enumerate(rpsArray):
            for jndex, J in enumerate(JArray):
                TMap[index, jndex, hndex] = propCt(J) * Density(h) * i ** 2\
                    * plant_dict['D'] ** 4
                
    powerMap = np.zeros((20, 20, 20))

    propCpFun = interpolate.interp1d(plant_dict['JIndex'], plant_dict['propCpData'],
                                     fill_value='extrapolate')
    
    for hndex, h in enumerate(hArray):
        for index, i in enumerate(rpsArray):
            for jndex, J in enumerate(JArray):
                powerMap[index, jndex, hndex] = propCpFun(J) * Density(h) * i ** 3 \
                    * plant_dict['D'] ** 5


def PlotTMap(PP, JArray, rpsArray):
    X, Y = np.meshgrid(JArray, rpsArray)
    TMapFig = plt.figure(1)
    TMapPlot = TMapFig.add_subplot(111, projection='3d')
    TMapPlot.plot_surface(X, Y, PP['TMap'][:, :, 0], cmap='summer')
    TMapPlot.set_xlabel('Advance ratio J ')
    TMapPlot.set_ylabel('Shaft Rotation (rev/s)')
    TMapPlot.set_zlabel('Thrust (N)')
    plt.title('Sea level Propeller Thrust')


def PlotPowerMap(PP, JArray, rpsArray):
    X, Y = np.meshgrid(JArray, rpsArray)
    powerMapFig = plt.figure(2)
    powerMapPlot = powerMapFig.add_subplot(111, projection='3d')
    powerMapPlot.plot_surface(X, Y, PP['powerMap'][:, :, 0], cmap='summer')
    powerMapPlot.set_xlabel('Advance ratio J ')
    powerMapPlot.set_ylabel('Shaft Rotation (rev/s)')
    powerMapPlot.set_zlabel('Power (W)')
    plt.title('Sea Level Propeller Shaft Power')
    plt.show()
