import numpy as np
import matplotlib.pyplot as plt
import math
import numbers
from scipy.optimize import minimize, Bounds
from matplotlib.ticker import StrMethodFormatter

from flight_operator import AnalyticalFindTP, TakeOff, Cruise, Climb, Glide, Hover, Drop
from simulator import simulate
from support_funs import GlideDistance, ClCd, PropTFun, GetCd, PlantSizing, AC
from get_params import Density, GetParams
from plot import Plot, PlotResiduals


def CostvsHeight(x=[20, 1.5, 1, 25], rng=50, plot=False):
    costArray = np.zeros(5)
    hArray = np.linspace(1, 5, 5)

    for index, i in enumerate(hArray):
        x[2] = i
        costArray[index] = CostF(x, rng)
    
    if plot:
        s = 'Climb: v1 = ' + str(climbv1) + ', v2 = ' + str(climbv2) + 'm/s'\
            + '\nCruise v = ' + str(cruisev1) + 'm/s' \
            + '\nRange = ' + str(rng) + 'km'
        y = (costArray[-1] - costArray[0]) * 0.85 + costArray[0]

        plt.plot(hArray, costArray)
        plt.xlabel('Cruise Altitude (km)')
        plt.ylabel('Mass Cost (kg)')
        plt.title('Mass Cost of Flight Profile vs. Cruise Altitude')
        plt.text(hArray[0], y, s)
        plt.show()
    return hArray, costArray
 
       
class Optimize(object):
    def __init__(self, rng=50, plant=None):
        self.xHist = np.zeros((4, 1))
        self.cHist = np.zeros(1)
        
        # if plant is not given, use standard unsized plant
        if not plant:
            self.plant = GetParams()[1]
        else:
            self.plant = plant
        
        self.rng = rng
        
        self.iterations = 0
        self.result = None
        
        self.x0 = np.array([10, 2.9, 5, 25])

        
    def CostF(self, x, fullOutput=False, saveSim=False):
        if len(x) != 4:
            raise ValueError('the cost function must be passed a 4 dimensional array')

        climbv1, climbv2, cruiseh, cruisev1 = x
    
        if self.iterations > len(self.xHist[0, :]) - 1:
            self.xHist = np.append(self.xHist, np.zeros((4, 10**2)), axis=1)
            self.cHist = np.append(self.cHist, np.zeros(10**2))
    
        self.xHist[0, self.iterations] = climbv1
        self.xHist[1, self.iterations] = climbv2
        self.xHist[2, self.iterations] = cruiseh
        self.xHist[3, self.iterations] = cruisev1
        
        # convert back to meters
        cruiseh *= 1000
        
        cs = simulate(self.rng, self.plant, runSizing=False)
    
        TakeOff(cs, fullOutput=fullOutput)
    
        Climb(cs, climbv1, climbv2, cruiseh, fullOutput=fullOutput)
    
        # find how long the cruise has to be
        cruiseDistance = self.rng * 1000 - cs.x1[cs.n] - GlideDistance(cruiseh)
        if cruiseDistance / cruisev1 > 15:
            Cruise(cs, cruisev1, cruiseDistance, fullOutput=fullOutput)
    
    
        Glide(cs, 0, fullOutput=fullOutput)
        
        cs.CalculateE()
        
        self.cHist[self.iterations] = cs.massCost[cs.n - 1]
        
        self.iterations += 1
        if self.iterations % 10 == 0:
            print(self.iterations)
            
        if saveSim:
            self.sim = cs
        
        return cs.massCost[cs.n - 1]


    def optm(self):
        self.iterations = 0
        
        x0 = np.array([16, 1, 5, 17])
        # x0 = np.array([20, 2, 3, 10])
    
        
        
        lower = [10, 0.5, 0.5, 10]
        upper = [25, 3, 5, 40]
        
        B = Bounds(lower, upper, keep_feasible=True)
        
        B = ((10, 25), (0.5, 3), (0.4, 5), (10, 40))
        
        # none - 875
        # 0.5 - iter = 250 - cost = 0.2861
        # 1 - iter = 235 - cost = 0.2863
        # 1.5 - iter = 160 - cost = 0.2925
        # 2 - iter = 160 - cost = 0.2925
        op = {'xtol': 1, 'barrier_tol': 1e-4}
        
        # nhcon = NonlinearConstraint(HConstraint, 0, np.inf)
        # vcon = NonlinearConstraint(VConstraint, 0, np.inf)
    
        self.result = minimize(self.CostF, x0, method='trust-constr',
                               options=op, bounds=(B))
        # a = basinhopping(CostF, x0, minimizer_kwargs={'args': desiredRange})
        
        print(self.result.nfev)
        return self.result


# improve this function's operation
# try getting the optimiser to be faster by changing the radius parameter
def Results(rng=50):
    global op, op1, op2, op3

    _, plant0 = GetParams()

    plant0 = PlantSizing(plant0)

    # save unchanged simulator for later
    s0 = simulate(rng, plant0, runSizing=False)

    ########## Start 1st opt ##########

    # create a sized plant dictionary
    _, plant1 = GetParams()
    plant1 = PlantSizing(plant1)

    # get prelim results with sized plant
    op1 = Optimize(rng, plant1)
    op1.optm()
    
    # save the simulator object that contains the last iteration of the optimisation
    op1.CostF(op1.result.x, saveSim=True)

    # amount of fuel actually needed for flight
    usedFuelM1 = op1.sim.fuelMassCost[op1.sim.n - 1]

    # necessary batter capacity needed for flight
    usedBattE1 = op1.sim.battE[op1.sim.n - 1]
    
    battChoice1 = 4
    for index, i in enumerate(plant1['battCapList']):
        if i > usedBattE1:
            battChoice1 = index
            break

    print('\nFuel mass used on the first optimisation:', round(usedFuelM1, 3), 'kg')
    print('Battery energy used on the first optimisation:', round(usedBattE1 / 1000, 3), 'kJ')
    print('Now attempting to fly with the', s0.PP['battMassList'][battChoice1], 'kg battery\n')

    ########## End 1st opt ##########

    ########## Start 2nd opt ##########
    # create another sized plant dictionary
    _, plant2 = GetParams()
    plant2 = PlantSizing(plant2)

    # update the masses, only change the plant2 dict to keep track of old values
    plant2['fullTankMass'] = usedFuelM1
    plant2['battChoice'] = battChoice1

    # get prelim results with sized plant
    op2 = Optimize(rng, plant2)
    op2.optm()

    # save new, lighter profile
    op2.CostF(op2.result.x, saveSim=True)

    # amount of fuel actually needed for flight
    usedFuelM2 = op2.sim.fuelMassCost[op2.sim.n - 1]
    # necessary battery capacity needed for flight
    usedBattE2 = op2.sim.battE[op2.sim.n - 1]

    # check if the same battery should be used
    for index, i in enumerate(plant2['battCapList']):
        if i > usedBattE2:
            battChoice2 = index
            break

    # check how much less fuel is used on second flight
    fSavings = abs((usedFuelM1 - usedFuelM2) / usedFuelM1)

    print('\nFuel mass used on the second optimisation:', round(usedFuelM2, 3), 'kg')
    print('Battery energy used on the second optimisation:', round(usedBattE2 / 1000, 3), 'kJ\n')

    ########## End Second opt ##########

    ########## Start 3rd opt ##########

    # if the fuel savings are siginificant again, or a different battery can be chosen
    # run the optimisation again
    if fSavings > 0.1 or battChoice2 != battChoice1:
        print('Now attempting to fly with the', s0.PP['battMassList'][battChoice2], 'kg battery')
        _, plant3 = GetParams()
        plant3 = PlantSizing(plant3)

        # update the masses, only change the s3 plant dict to keep track of values
        
        plant3['fullTankMass'] = usedFuelM2
        plant3['battChoice'] = battChoice2
        
        # run optimization again but with smaller battery etc.
        op3 = Optimize(rng, plant3)
        op3.optm()
        
        # save new profile
        op3.CostF(op3.result.x, saveSim=True)
        
        battMassSavings = s0.battMass - op3.sim.battMass
        fuelMassSavings = s0.PP['fullTankMass'] - op3.sim.PP['fullTankMass']
    
    ########## End 3rd opt ##########

    # if second run was sufficient, should get values for saved mass
    else:
        battMassSavings = s0.battMass - op2.sim.battMass
        fuelMassSavings = s0.PP['fullTankMass'] - op2.sim.PP['fullTankMass']
    
    print('The optimisation terminated. The optimisation saved at least', round(battMassSavings, 2),
          'kg in battery mass and ', round(fuelMassSavings, 2),
          'kg in fuel which are now available', 'for payload')

    if 's3' in locals():
        return op1, op2, op3
    
    else:
        return op1, op2


def Fly():
    global sim, fly

    _, PP = GetParams()
    
    # initialize classes
    sim = simulate(700, PP)
    
    # take off
    TakeOff(sim)

    Climb(sim, v1=13.66, desiredCR=2, desiredHeight=410)

    Cruise(sim, v1=16.9, distance=500)

    Glide(sim, 0)
    
    sim.CalculateE()
    
    # Plot(sim, 'x2', 'v1', 'v2', 'angles', 'thrust', 'power')


def PlantResults():
    global plantS
    plantS = simulate(1000)
    
    TakeOff(plantS)
    Climb(plantS, v1=16, desiredCR=1, desiredHeight=100)
    Climb(plantS, v1=16, desiredCR=3, desiredHeight=200)
    Cruise(plantS, v1=18, distance=2000)
    plantS.CalculateE()
    
    Plot(plantS, 'x2', 'v1v2', 'energy', 'power', 'speed', 'soc', 'thrust',
         'pitch', 'plant', 'iceeff', markers=False, save=True, title=False,
         direct='simulationPics/plantResults/')
    
def DropandHover():
    global s
    
    PP = GetParams()[1]
    
    s = simulate(500)
    
    initialHeight = 500
    
    s.y0 = [0, initialHeight, 0, 0]
    s.x2[s.n] = initialHeight
    s.pitch[s.n] = 90

    s.pitchShape = 'const'
    s.pitchCeiling = 90
    
    
    # make the aircraft mass lower because the powerplant is not enough to
    # make the full aircraft hover
    s.totalMass = 4

    weight = s.totalMass * AC['g']

    s.TShape = 'sigmoid'
    s.TFloor = 0
    s.TCeiling = weight
    s.TStart = 0
    s.TEnd = 50
    
    s.runtime = 200
    s.steps = 200
    s.RunInterval()   
    
    Plot(s, 'thrust', 'pitch', 'x2', 'v2', save=True, title=False, direct='simulationPics/hover/')
    
    Fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(s.timeArray[:s.n], s.x2[:s.n], label='Height')
    ax2.plot(s.timeArray[:s.n], s.v2[:s.n], 'r', label='V2')
    
    ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.set_xlabel('time (seconds)')
    ax1.set_ylabel('AGL (m)')
    ax2.set_ylabel('velocity (m/s)')
    
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.tight_layout()
    plt.savefig('simulationPics/hover/x2v2.png')
    plt.show()

    
def TestCR():
    s = simulate(50000)

    # take off
    TakeOff(s)

    Climb(s, v1 = 20, desiredCR = 1, desiredHeight=100)
    
    crEstimate = GetExpectedCR(s)
    
    print('Excess power estimate for the climb rate:', crEstimate, 'm/s')
    
    Plot(s, 'thrust', 'pitch', 'v1andv2', title=False, save=True, direct='simulationPics/testCR/', markers=False)
    

    """
    Fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(s.timeArray[:s.n], s.x2[:s.n], label='Height')
    ax2.plot(s.timeArray[:s.n], s.v1[:s.n], 'r', label='V1')
    
    ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.set_xlabel('time (seconds)')
    ax1.set_ylabel('height (m)')
    ax2.set_ylabel('velocity (m/s)')
    
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.tight_layout()
    plt.savefig('simulationPics/testCR/x2v1.png')
    plt.show()
    """

def GetExpectedCR(s):
    """This is the excess power estimate for climb rate, returns in m/s """
    weight = s.totalMass * AC['g']
    
    drag = 0.5 * Density(s.x2[s.n]) * AC['S'] * GetCd(s.alpha[s.n], s.x2[s.n]) * s.V() ** 2
    
    CR = s.V() * (s.thrust[s.n] - drag) / weight
    
    return CR        


def MassCost(s, op):
    power = s.PowerCost(op)

    batteryMassCost = power[0] * op.ts / batteryDensity
    fuelCost = power[1] * op.ts / LCV

    return batteryMassCost, fuelCost

def FindConstraints():
    """
    Calculates the maximum horizontal and vertical velocities and height which the power-plant
    can achieve. These are stored in a global 3x30 array called limits

    Returns
    -------
    None.

    """
    global limits
    print(['limits' in globals()])
    print(['limits' in locals()])

    limits = np.zeros((3, 15**2))  # (maxv1, maxv2, maxHeight)
    
    print(['limits' in globals()])
    print(['limits' in locals()])
    
    v1Array = np.linspace(10, 50, 15)
    hArray = np.linspace(100, 6000, 15)
    
    s = simulate(10e7)
    counter = 0
    for hndex, h in enumerate(hArray):
        for index, i in enumerate(v1Array):
        
            T = 0
            maxThrust = 1
            v2 = 0
            while T < maxThrust:
                v2 += 0.5
                v = math.sqrt(i ** 2 + v2 ** 2)
                maxThrust = float(PropTFun(v, PP['ICEMaprps'][-1], h) + PropTFun(v, PP['maxEMrps'], h))
                T = float(AnalyticalFindTP(s, h, i, v2)[0])
    
            limits[0, counter] = i
            limits[1, counter] = v2 - 0.5
            limits[2, counter] = h
            
            counter += 1
    
    return limits
    

def ConstraintFun(v1, h):
    """
    Uses the constraints in the limits array to find the maximum v2 value for a given v1 and h
    that the powerplant can achieve. Essentially this function finds the data point in limits
    associated with the given v1 and h.

    Parameters
    ----------
    v1 : int or float
        Horizontal velocity.
    h : int or float
        Altitude.

    Raises
    ------
    ValueError
        When wrong arg type is passed

    Returns
    -------
    np.array
       triplet containing the limiting v1, v2 and h values
    
    """
    global indexV1Bool, indexHBool, limitH, limitV1, v1Index, hIndex, index
    
    if not isinstance(v1, numbers.Number):
        raise ValueError('Wrong arg type passed to ConstraintFun() for v1. Must be'
                         + ' int or float, not ' + str(type(v1)))
    
    if not isinstance(h, numbers.Number):
        raise ValueError('Wrong arg type passed to ConstraintFun() for h. Must be'
                         + ' int or float, not ' + str(type(v1)))
        


    v1Index = np.argmin(abs(v1 - limits[0]))
    hIndex = np.argmin(abs(h - limits[2]))
    
    limitV1 = limits[0, v1Index]
    limitH = limits[2, hIndex]
    
    # array that contains 'True' at all the positions where limitV1 can be
    # found
    indexV1Bool = [i == limitV1 for i in limits[0]]
    
    # same as above but for limitH
    indexHBool = [i == limitH for i in limits[2]]
            
    
    v1List = [limitV1 for i in limits[0] if i==limitV1]
    
    # array that has an index in the right position
    index = [index if (i and j) else False for index, (i, j) in
             enumerate(zip(indexV1Bool, indexHBool))]

    # cut all the unnecessary values from the array
    index = list(filter(lambda x: type(x) != bool, index))
    
    # print(limits[:, index])
    return limits[:, index]
    
def PlotConstraints():
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(limits[0], limits[1], limits[2])
    ax1.set_xlabel('Horizontal Velocity (m/s)')
    ax1.set_ylabel('Climb Rate (m/s)')
    ax1.set_zlabel('Altitude (m)')


def PlotLevelFlightV():
    """Plots the level flight velocity of the aircraft against angle of
    attack"""
    alphaArray = np.linspace(-maxAlpha, maxAlpha, 50)
    vArray = np.zeros(50)
    sim = simulate()
    sim.x2[sim.tost-1] = 0
    for i in range(len(alphaArray)):
        sim.alpha = alphaArray[i]
        vArray[i] = GetLevelFlightV(sim)
        
    plt.plot(alphaArray, vArray)
    plt.xlabel('angle of attack in degrees')
    plt.ylabel('level flight velocity (m/s)')
    plt.title('Level Flight Velocity at different angles of attack')
    plt.savefig('simulationPics/levelFlight.png')

