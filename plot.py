import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

from support_funs import Sigmoid, AC, GetCd, PlantSizing
from get_params import GetParams

# use the style for plots
try:
    plt.style.use('ipstyle')
except:
    pass


def PlotResiduals(op, plotC=True, plotX=True):
    """
    Plots the residuals and cost function values from an optimization object

    Parameters
    ----------
    op : optimization object
    plotC : bool, optional
        Determine whether to plot the cost function. The default is True.
    plotX : TYPE, optional
        Determine whether to plot the residuals. The default is True.

    Returns
    -------
    None.

    """
    
    x = op.result.x

    iterArray = np.linspace(0, op.iterations - 1, op.iterations)

    if plotX:
        resFig, resAx1 = plt.subplots()
        resAx1.plot(iterArray, op.xHist[0, :op.iterations] - x[0], label='Climb V1')
        resAx1.plot(iterArray, op.xHist[1, :op.iterations] - x[1], label='Climb V2')
        resAx1.plot(iterArray, op.xHist[2, :op.iterations] - x[2], label='Cruise Alt')
        resAx1.plot(iterArray, op.xHist[3, :op.iterations] - x[3], label='Cruise V1')
        resAx1.set_xlabel('number of iterations')
        resAx1.set_ylabel('Residuals')
        
        if plotC:
            resAx2 = resAx1.twinx()
            resAx2.plot(iterArray, op.cHist[:op.iterations], color='k', label='Solution')
            lines2, labels2 = resAx2.get_legend_handles_labels()
            lines, labels = resAx1.get_legend_handles_labels()
            resAx2.set_ylabel('Solution')
    
            resAx2.legend(lines + lines2, labels + labels2, loc=0)
        else:
            plt.legend()
    
    else:
        plt.plot(iterArray, op.cHist[:op.iterations], color='k', label='Solution')
        plt.xlabel('number of iterations')
        plt.ylabel('equivalent mass cost')
    
    plt.title('')
    plt.show()



def PlotPlantSizing():
    """
    Plots the power plant mass (ICE mass + EM mass) produced when sizing at different
    rEM. rEM = maximum EM power / total power demand as given by ADRpy

    Returns
    -------
    None.

    """
    _, PP = GetParams()
    
    rEMArray = np.linspace(0, 1, 11)
    massArray = np.zeros(11)
    
    for index, i in enumerate(rEMArray):
        sizedP = PlantSizing(PP, i)
        m = sizedP['EMMass'] + sizedP['ICEMass']
        massArray[index] = m
        
    plt.plot(rEMArray, massArray)
    plt.xlabel('Ratio of power produced by EM')
    plt.ylabel('ICE + EM mass (kg)')
    plt.title('Plant mass at different ICE and EM sizes')


def PlotSigmoid():
    """ Produces some plots to demonstrate the Sigmoid function's ability """

    global sigArray
    t = np.linspace(-5, 15, 50)
    for i in range(5):
        sigArray = np.zeros(50)
        
        ceiling = (i + 1) * 3
        floor = i
        start = -3 + i
        end = i * 3
        
        sigArray = Sigmoid(t, floor, ceiling, start, end)
        
        lab = 'ceiling= ' + str(ceiling) + ' floor= ' + str(floor) + ' start= ' \
            + str(start) + ' end= ' + str(end)
        plt.plot(t, sigArray, label=lab)
        
    plt.legend()
    plt.title('Range of sigmoid functions')
    # plt.savefig('simulationPics/sigmoids.png')
    plt.show()



def PlotICEMap(PP):
    """ Produces a surface plot of the ICE efficiency map from the .txt file """
    X, Y = np.meshgrid(PP['ICEMaprps'], PP['ICEMapTorque'])
    
    effFig = plt.figure()
    effPlot = effFig.add_subplot(111, projection='3d')
    effPlot.plot_surface(X, Y, PP['effMap'], cmap='summer')
    effPlot.set_xlabel('Shaft rotational speed (rev/s)')
    effPlot.set_ylabel('Torque (Nm)')
    effPlot.set_zlabel('Efficiency')
    effPlot.set_title('ICE Efficiency Map')


def PlotDragPolar():
    clArray = np.linspace(-1, 1, 40)
    cdArray = np.array([])
    for i in clArray:
        # print('Cl ' + str(i))
        Cd = AC['parasiteDrag'] + i ** 2 / (AC['AR'] * AC['e'] * math.pi)
        cdArray = np.append(cdArray, Cd)
        # print('Cd ' + str(Cd) +'\n')

    plt.plot(clArray, cdArray)
    plt.xlabel('Coefficient of Lift')
    plt.ylabel('Coefficient of Drag')
    plt.title('Drag Polar')


def PlotGroundEffect():
    cdArray = np.zeros(20)
    hArray = np.linspace(0, 4, 20)
    alphaArray = np.linspace(0, 10, 5)
    
    for index, i in enumerate(alphaArray):
        for jndex, j in enumerate(hArray):
            cdArray[jndex] = GetCd(alphaArray[index], hArray[jndex])
            
        plt.plot(hArray, cdArray, label='angle of attack =' + str(alphaArray[index]))
    
    plt.xlabel('Height above the ground')
    plt.ylabel('Coefficient of Drag')
    plt.legend()
    plt.show()


def PlotPropeller(PP):
    # https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19930081563.pdf
    # 2 blade 25 degrees prop
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(PP['JIndex'], PP['propEffData'] * 100, 'r', label='Efficiency')
    ax1.set_xlabel('Advance Ratio (J)')
    ax1.set_ylabel('Efficiency (%)')
    
    ax2 = ax1.twinx()
    ax2.plot(PP['JIndex'], PP['propCpData'], label='Coefficient of Power')
    ax2.plot(PP['JIndex'], PP['propCtData'], label='Coefficient of Thrust')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax2.set_ylabel('Thrust and Power Coefficient')
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    # plt.title('Propeller data over advance ratio')
    plt.show()
    

def Plot(s, *args, save=False, timerange='all', title=True, direct='simulationPics/', markers=True):
    """
    Plot flight data from simulator object. *args allows plotting any combination of data or
    all data

    Parameters
    ----------
    s : simulate OBJECT
        simulator object needed to obtain data
    *args : str
        Indicate which data to plot.
    save : bool, optional
        If true, plots are saved at the specified (or default) directory. The default is False.
    timerange : tuple, optional
        Time values between which the plot is to be shown. The default is 'all'.
    title : bool, optional
        If TRUE, the plots will have titles. The default is True.
    direct : str, optional
        Directory for saving the plots. The default is 'simulationPics/'.
    markers : bool, optional
        If TRUE, markers are plotted at every instance where the simulate.RunInterval
        method was called. The default is True.


    Raises
    ------
    ValueError
        When wrong arg type is passed.

    Returns
    -------
    None.

    """


    start = 0
    end = s.n

    # get the steps at which the required times occured, within some margin
    # because the exact required timestep might not be required
    if timerange != 'all':
        initialTime = timerange[0]
        finalTime = timerange[1]
        start = np.argmin(abs(initialTime - s.timeArray))
        end = np.argmin(abs(finalTime - s.timeArray))

    # set the x axis by cutting the simulator time array according to
    # user request. Set the units by converting to minutes if the
    # time is very long
    if s.timeArray[end] - s.timeArray[start] > 500:
        timeLabel = 'time (minutes)'
        tTemp = s.timeArray[start:end] / 60

    else:
        timeLabel = 'time (seconds)'
        tTemp = s.timeArray[start:end]


    # special keywords that can be passed to args to plot groups of plots
    allAngles = ('alpha', 'gamma', 'pitch')
    kin = ('x1', 'x2', 'v1', 'v2', 'a1', 'a2')

    if 'all' in args:
        args = kin + ('T', 'power', 'energy', 'SOC', 'total efficiency', 'plant') + allAngles

    elif any(s in args for s in ['angles', 'angle']):
        if type(args) == tuple:
            args += tuple(allAngles)
        else:
            args += allAngles

    elif 'kin' in args: args += kin

    elif 'vertical' in args: args += ('x2', 'v2', 'a2')

    elif 'horizontal' in args: args += ('x1', 'v1', 'a1')

    # prepare RunInterval start markers
    if markers:
        m = np.ndarray.tolist(s.plotMarkers[s.plotMarkers < len(tTemp)])
        m = [int(x) for x in m]
        # marker shape
        D = '-D'
        rD = '-rD'
        gD = '-gD'
    else:
        m = np.nan
        D = ''
        rD = 'r'
        gD = 'gD'
        
    args = [i.lower() for i in args]

    if any(s in args for s in ['rps', 'speed']):
        x = np.linspace(tTemp[0], tTemp[-1], 10)
        y = np.repeat(s.PP['idealICErps'], len(x))
        
        # rpsFig, rpsAx1 = plt.subplots(figsize=(9, 5))
        rpsFig, rpsAx1 = plt.subplots()
        rpsAx1.plot(x, y, 'r', label='ideal ICE speed')
        rpsAx1.plot(tTemp, s.ICErps[start:end], D, markevery=m, label='ICE speed')
        rpsAx1.plot(tTemp, s.EMrps[start:end], D, markevery=m, label='EM speed')
        if title: plt.title('ICE and EM shaft speed')
        rpsAx1.set_xlabel(timeLabel)
        rpsAx1.set_ylabel('Shaft Speed (rev/s)')
        rpsAx1.legend()
        if save: plt.savefig(direct + 'rps.png')
        plt.show()

    if any(s in args for s in ['total efficiency']):
        plt.plot(tTemp, s.EMTotEff[start:end] * 100, rD, markevery=m, label='Total EM efficiency')
        plt.plot(tTemp, s.ICETotEff[start:end] * 100, gD, markevery=m, label='Total ICE efficiency')
        if title: plt.title('Overall Powerplant efficiency over time')
        plt.xlabel(timeLabel)
        plt.ylabel('efficiency (%)')
        plt.legend()
        if save: plt.savefig(direct + 'totEff.png')
        plt.show()
    
    if any(s in args for s in ['plant', 'powerplant']):
        a = 0.4  # transparency value for background
        PlantFig, PlantAx1 = plt.subplots(figsize=(9, 5))
        PlantAx2 = PlantAx1.twinx()

        PlantAx1.fill_between(tTemp, s.RBOut[start:end] == 0, color='lightgrey',
                              label='plant off', step='mid', alpha=a)
        PlantAx1.fill_between(tTemp, s.RBOut[start:end] == 1, color='green',
                              label='ICE only', step='mid', alpha=a)
        PlantAx1.fill_between(tTemp, s.RBOut[start:end] == 2,
                              label='ideal ICE + EM', step='mid', alpha=a)
        PlantAx1.fill_between(tTemp, s.RBOut[start:end] == 3,
                              label='max EM + ICE', step='mid', alpha=a)
 
        PlantAx2.plot(tTemp, s.EMThrust[start:end], D, color='blue', markevery=m, label='EM Thrust')
        PlantAx2.plot(tTemp, s.ICEThrust[start:end], D, color='k', markevery=m, label='ICE Thrust')
        PlantAx1.plot(tTemp, s.rEM[start:end], rD, markevery=m, label='Share of EM Thrust')
   
        lines, labels = PlantAx1.get_legend_handles_labels()
        lines2, labels2 = PlantAx2.get_legend_handles_labels()

        PlantAx1.set_xlabel(timeLabel)
        PlantAx2.set_ylabel('Thrust (N)')
        PlantAx1.set_ylabel('Share of EM Thrust')
        
        PlantAx2.legend(lines + lines2, labels + labels2, loc=0)

        if save: plt.savefig(direct + 'plant.png')
        if title: plt.title('EM, ICE thrust and power distribution')
        plt.show()
    
    # horizontal displacement
    if 'x1' in args:
        plt.plot(tTemp, s.x1[start:end], D, markevery=m)
        plt.xlabel(timeLabel)
        plt.ylabel('Horizontal Distance (m)')
        if title: plt.title('Horizontal Distance over Time')
        if save: plt.savefig(direct + 'x1.png')
        plt.show()
        
    # horizontal velocity
    if 'v1' in args:
        plt.plot(tTemp, s.v1[start:end], D, markevery=m)
        plt.xlabel(timeLabel)
        plt.ylabel('Horizontal Velocity (m/s)')
        if title: plt.title('Horizontal Velocity over Time')
        if save: plt.savefig(direct + 'v1.png')
        plt.show()
        
    # horizontal acceleration
    if 'a1' in args:
        plt.plot(tTemp, s.a1[start:end], D, markevery=m)
        plt.xlabel(timeLabel)
        plt.ylabel('Horizontal Acceleration (m/s2)')
        if title: plt.title('Horizontal Acceleration over Time')
        if save: plt.savefig(direct + 'a1.png')
        plt.show()
        
    # altitude
    if 'x2' in args:
        plt.plot(tTemp, s.x2[start:end], D, markevery=m)
        plt.xlabel(timeLabel)
        plt.ylabel('AGL (m)')
        if title: plt.title('Height over Time')
        if save: plt.savefig(direct + 'x2.png')
        plt.show()
        
    # vertical velocity
    if 'v2' in args:
        plt.plot(tTemp, s.v2[start:end], D, markevery=m)
        plt.xlabel(timeLabel)
        plt.ylabel('Vertical Velocity (m/s)')
        if title: plt.title('Vertical Velocity over Time')
        if save: plt.savefig(direct + 'v2.png')
        plt.show()
    
    # vertical acceleration
    if 'a2' in args:
        plt.plot(tTemp, s.a2[start:end], D, markevery=m)
        plt.xlabel(timeLabel)
        plt.ylabel('Vertical Acceleration (m/s2)')
        if title: plt.title('Vertical Acceleration over Time')
        if save: plt.savefig(direct + 'a2.png')
        plt.show()
        
    # thrust signal, pitch signal, angle of attach, flight path angle or any combination of them
    if any(x in args for x in ('pitch', 'Pitch', 'alpha', 'Alpha', 'gamma', 'Gamma', 'T',
                               't', 'thrust', 'Thrust')):
        # angleFig, angleAx1 = plt.subplots(figsize=(9, 5))
        angleFig, angleAx1 = plt.subplots()
        t = ''
        if any(s in args for s in ['pitch', 'Pitch']):
            angleAx1.plot(tTemp, s.pitch[start:end], D, markevery=m, label='Pitch')
            t += 'Angles '
    
        if any(s in args for s in ['alpha', 'Alpha']):
            angleAx1.plot(tTemp, s.alpha[start:end], D, markevery=m, label='Angle of attack')
            if len(t) == 0: t += 'Angles '
    
        if any(s in args for s in ['gamma', 'Gamma']):
            angleAx1.plot(tTemp, s.gamma[start:end], D, markevery=m, label='Flight Path Angle')
            if len(t) == 0: t += 'Angles '
    
        if any(s in args for s in ['thrust']):
            # plot on secondary axis if angles are also plotted
            if len(t) != 0:
                t += 'and '
                angleAx2 = angleAx1.twinx()
                angleAx2.plot(tTemp, s.thrust[start:end], rD, markevery=m, label='Thrust')
                angleAx2.set_ylim(min(s.thrust[start:end]) * 1.1, max(s.thrust[start:end]) * 1.1)
                lines2, labels2 = angleAx2.get_legend_handles_labels()
                angleAx2.set_ylabel('Thrust (N)')
                
            # plot on primary axis if angles are not plotted
            else:
                angleAx1.plot(tTemp, s.thrust[start:end], rD, markevery=m, label='Thrust')
                angleAx1.set_ylabel('Thrust (N)')
                
            t += 'Thrust '
        
        angleAx1.set_xlabel(timeLabel)
        
        
        # only set angle label if angles are being plotted
        if 'Angles' in t: angleAx1.set_ylabel('Angle (degrees)')
        
        lines, labels = angleAx1.get_legend_handles_labels()
        
        # use this legend if angles AND thrust are plotted
        if 'and' in t:
            angleAx2.legend(lines + lines2, labels + labels2, loc=0)
        else:
            plt.legend()
        if title: plt.title(t)
        if save: plt.savefig(direct + 'tanda.png')
        plt.show()
        
        
    # SOC and fuel remaining
    if any(s in args for s in ['soc', 'charge', 'fuel', 'tank']):
        fig, ax1 = plt.subplots()
        ax1.plot(tTemp, 100 * s.SOC[start:end], D, markevery=m, label='SOC')
        ax1.set_xlabel(timeLabel)
        ax1.set_ylabel('SOC (%)')
        
        ax2 = ax1.twinx()
        ax2.plot(tTemp, s.fuelMass[start:end], rD, markevery=m, label='fuel remaining')
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.set_ylabel('fuel mass (kg)')
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        if title: plt.title('SOC and remaining fuel over time')
        if save: plt.savefig(direct + 'fuelandsoc.png')
        plt.show()
        
    # E used by ICE, EM and total
    if any(s in args for s in ['Energy', 'energy']):
        plt.plot(tTemp, s.fuelE[start:end] / 1000, D, markevery=m, label='E used by ICE')
        plt.plot(tTemp, s.battE[start:end] / 1000, D, markevery=m, label='E used by EM')
        plt.plot(tTemp, s.totalE[start:end] / 1000, D, markevery=m, label='Total E used')
        if title: plt.title('Energy used over time (cumulatively)')
        plt.xlabel(timeLabel)
        plt.ylabel('Energy (kJ)')
        plt.legend()
        if save: plt.savefig(direct + 'energy.png')
        plt.show()

    if any(s in args for s in ['mass', 'cost']):
        mFig, mAx1 = plt.subplots()
        mAx1.fill_between(tTemp, s.massCost[start:end], s.battMassCost[start:end],
                          label='Fuel')
        mAx1.fill_between(tTemp, s.battMassCost[start:end], label='Battery')
        plt.xlabel(timeLabel)
        plt.legend()
        if title: plt.title('Accumulated mass cost')
        if save: plt.save(direct + 'massCost.png')

    # power used by ICE, EM and total
    if any(s in args for s in ['Power', 'power']):
        plt.plot(tTemp, s.battP[start:end] / 1000, D, markevery=m, label='P drawn by EM')
        plt.plot(tTemp, s.fuelP[start:end] / 1000, D, markevery=m, label='P drawn by ICE')
        plt.plot(tTemp, s.power[start:end] / 1000, D, markevery=m, label='Power output')
        if title: plt.title('Power over time')
        plt.xlabel(timeLabel)
        plt.ylabel('Power (kW)')
        plt.legend()
        if save: plt.savefig(direct + 'power.png')
        plt.show()
        
    if any(s in args for s in ['iceeff', 'engine eff']):
        plt.plot(tTemp, s.ICEEff[start:end] * 100)
        plt.xlabel(timeLabel)
        plt.ylabel('efficiency (%)')
        if title: plt.title('ICE brake efficiency over time')
        if save: plt.savefig(direct + 'ICEEff')
        plt.show()
        
    # shaft power of ICE, EM and total
    if any(s in args for s in ['engine', 'ICE', 'ice']):
        ICEFig, ICEAx1 = plt.subplots()
        ICEAx2 = ICEAx1.twinx()
        ICEAx1.plot(tTemp, s.ICEShaftP[start:end], D, markevery=m, label='Shaft Power')
        ICEAx1.plot(tTemp, s.ICEPropP[start:end], D, markevery=m, label='Propeller Power')
        ICEAx2.plot(tTemp, s.fuelMass[start:end], rD, markevery=m, label='Fuel left')
        
        lines, labels = ICEAx1.get_legend_handles_labels()
        lines2, labels2 = ICEAx2.get_legend_handles_labels()
        
        ICEAx1.set_xlabel('timeLabel')
        ICEAx1.set_ylabel('Power (W)')
        ICEAx2.set_ylabel('Fuel Mass (kg)')
        
        ICEAx2.legend(lines + lines2, labels + labels2, loc=0)
        
        if title: plt.title('ICE Power and fuel level over time')
        if save: plt.savefig(direct + 'ICE.png')
        plt.show()
        
    # EM shaft and used power and SOC
    if any(s in args for s in ['motor', 'Motor', 'EM', 'em']):
        EMFig, EMAx1 = plt.subplots()
        EMAx2 = EMAx1.twinx()
        EMAx1.plot(tTemp, s.EMShaftP[start:end], D, markevery=m, label='Shaft Power')
        EMAx1.plot(tTemp, s.EMPropP[start:end], D, markevery=m, label='Propeller Power')
        EMAx2.plot(tTemp, s.SOC[start:end] * 100, rD, markevery=m, label='State of Charge')
        
        lines, labels = EMAx1.get_legend_handles_labels()
        lines2, labels2 = EMAx2.get_legend_handles_labels()
        
        EMAx1.set_xlabel(timeLabel)
        EMAx1.set_ylabel('Power (W)')
        EMAx2.set_ylabel('SOC (%)')
        
        EMAx2.legend(lines + lines2, labels + labels2, loc=0)
        
        if title: plt.title('EM Power and SOC over time')
        if save: plt.savefig(direct + 'SOC.png')
        plt.show()
        
    if any(s in args for s in ['v1andv2', 'v1v2']):
        # v1v2Fig, v1v2Ax1 = plt.subplots(figsize=(9, 5))
        v1v2Fig, v1v2Ax1 = plt.subplots()
        v1v2Ax2 = v1v2Ax1.twinx()
        v1v2Ax1.plot(tTemp, s.v1[start:end], label='v1')
        v1v2Ax2.plot(tTemp, s.v2[start:end], color='tab:orange', label='CR')
        lines, labels = v1v2Ax1.get_legend_handles_labels()
        lines2, labels2 = v1v2Ax2.get_legend_handles_labels()
        v1v2Ax1.set_ylim(min(s.v1[start:end]) * 1.1, max(s.v1[start:end]) * 1.1)
        v1v2Ax1.set_xlabel(timeLabel)
        v1v2Ax1.set_ylabel('horizontal velocity (m/s)')
        v1v2Ax2.set_ylabel('climb rate (m/s)')
        v1v2Ax2.legend(lines + lines2, labels + labels2, loc=0)
        if title: plt.title('Horizontal and Vertical V over Time')
        if save: plt.savefig(direct + 'v1v2.png')
        plt.show()
    
    """
    if any(s in args for s in []]):
        Fig, Ax1 = plt.subplots()
        x2 = Ax1.twinx()
        x1.plot(tTemp, , '-D', markevery=m, label='')
        x1.plot(tTemp, , '-D', markevery=m, label='')
        x2.plot(tTemp, , '-rD', markevery=m, label='')
        
        lines, labels = x1.get_legend_handles_labels()
        lines2, labels2 = x2.get_legend_handles_labels()
        
        x1.set_xlabel('timeLabel')
        x1.set_ylabel('')
        x2.set_ylabel('')
        
        x2.legend(lines + lines2, labels + labels2, loc=0)
        
        plt.title('')
        plt.show()
    
    """
    """
    if any(s in args for s in []):
        plt.plot(tTemp, , '-D', markerevery=m)
        plt.title('')
        plt.xlabel(timeLabel)
        plt.ylabel('')
        plt.show()
        
    """