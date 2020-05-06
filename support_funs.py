""" contains some physics to use in the simulaotr """

import matplotlib.pyplot as plt
import numbers
import math
import numpy as np
from scipy.optimize import minimize_scalar, fsolve
from scipy.interpolate import interp2d

from ADRpy import atmospheres as at
from ADRpy import constraintanalysis as ca
from ADRpy import unitconversions as co

from get_params import GetParams, Density


def VLO(s, alpha=0):
    """
    Lift off speed as 1.2 * stall velocity. (McCormick)

    Parameters
    ----------
    s : simulate OBJECT
        DESCRIPTION.

    Raises
    ------
    ValueError
        When wrong arg type is passed.

    Returns
    -------
    TYPE
        Estimate of liftoff velocity

    """

    if not hasattr(s, 'steps'):
        raise ValueError('Wrong arg type passed to VLO() for s. Must be'
                         + ' simulator object, not ' + str(type(s)))

    Clmax = GetCl(alpha)

    weight = s.totalMass * AC['g']
    vStall = math.sqrt(2. * weight / (AC['S'] * AC['rho0'] * Clmax))

    return 1.2 * vStall


def ZeroPitchCruiseV(s, alpha='none'):
    """
    Rearranged vertical EOM for horizontal velocity
    
    Parameters
    ----------
    s : simulat OBJECT.
    alpha : FLOAT, optional value to force the angle of attack. The default is simulator value.
    
    Raises
    ------
    ValueError
        When wrong arg types are passed.

    Returns
    -------
    v : FLOAT
        Horizontal velocity at which the aircraft will cruise at zero pitch.
    """

    if alpha == 'none':
        alpha = s.alpha[s.n]
    elif not isinstance(alpha, numbers.Number):
        raise ValueError('Wrong arg type passed to ZeroPitchCruiseV() for alpha. Must be'
                         + ' float or int')
    
    if not hasattr(s, 'steps'):
        raise ValueError('Wrong arg type passed to ZeroPitchCruiseV() for s. Must be'
                         + ' simulator object')

    Cl = GetCl(alpha)
    weight = s.totalMass * AC['g']
    v = math.sqrt(2. * weight / (AC['S'] * Density(s.x2[s.n]) * Cl))

    return v


def GetCl(alpha):
    """
    Linear model for the coefficient of lift as a function of alpha

    Parameters
    ----------
    alpha : float or int
        Angle of attack.

    Returns
    -------
    float
        Coefficient of lift at the given angle of attack.

    """
    
    return AC['clSlope'] * alpha + AC['clIntercept']



def GetCd(alpha, h):
    """
    Total coefficient of drag acting on aircraft. Includes ground effect.

    Parameters
    ----------
    alpha : int or float
        Angle of attack.
    h : int or float
        Height of the wing above the ground.

    Raises
    ------
    ValueError
        When args of wrong type are passed.

    Returns
    -------
    float
        Coefficient of drag.

    """

    if not isinstance(h, numbers.Number):
        raise ValueError('Wrong arg type passed to GetCd() for h. Must be'
                         + ' int or float')

    return AC['parasiteDrag'] + (16 * h / AC['span']) ** 2 / (1 + (16 * h / AC['span']) ** 2) * \
           GetCl(alpha) ** 2 / (AC['AR'] * AC['e'] * math.pi)



def ClCd(plot=False, h=1000):
    """
    Max Lift to Drag ratio and angle at which it occurs. Uses minimize_scalar
    over GetCl()/GetCd() to find this point.

    Parameters
    ----------
    plot : bool, optional
        If True, a plot of the L/D ratio over alpha is produced. The default is False.
    h : int or float, optional
        Height value. Only important if the ground effect on drag is relevant. The default is 1000.

    Raises
    ------
    ValueError
        When wrong arg type is passed.

    Returns
    -------
    float
        Angle of attack where the maximum L/D occurs in degrees.
    flaot
        Maximum L/D value.

    """
    
    if type(plot) is not bool:
        raise ValueError('Wrong arg type passed to ClCd() for plot. Must be'
                         + ' bool')
        
    if not isinstance(h, numbers.Number):
        raise ValueError('Wrong arg type passed to ClCd() for h. Must be'
                         + ' int or float')
        
        
    bestAlpha = minimize_scalar(lambda alpha: -GetCl(alpha) / GetCd(alpha, h))

    if plot:
        clcd = np.zeros(20)

        alphaArray = np.linspace(0, 20, 20)

        for index, i in enumerate(clcd):
            clcd[index] = GetCl(alphaArray[index]) / GetCd(alphaArray[index], 1000)
            
        plt.plot(alphaArray, clcd, markevery=bestAlpha.x)
        plt.title('Lift to drag ratio')
        plt.xlabel('angle of attack')
        plt.ylabel('L/D')
        plt.show()
            
    return bestAlpha.x, -bestAlpha.fun


def GlideDistance(deltaH):
    """
    Returns the distance that can be glided given an initial height. Assumes aircraft is
    pitched so that the maximum L/D occurs, which should be the case if using
    flight_operator.Glide

    Parameters
    ----------
    deltaH : int or float
        Height to be covered during the gliding descent.

    Returns
    -------
    float
        Approximate distance the glide will cover in meters.

    """
    
    return deltaH * ClCd()[1]


def ICEEff(PP, rps, Q):
    """Interpolate efficiency map using interpolation function from powerplant dictionary.
    Essentially adds some padding to the interpolation to prevent the ICE from running at
    excessive rps or Q

    Parameters
    ----------
    PP: dict
        Power plant dictionary
    rps : int or float
        ICE shaft speed.
    Q : int or float
        ICE shaft torque

    Raises
    ------
    ValueError
        If rps or Q parameters exceed the bounds of the ICE efficiency map. The ICE cannot
        operate outside these bounds.

    Returns
    -------
    float
        efficiency as a decimal.

    """
    
    # if the rps value is too high, give error message
    if rps > PP['ICEMaprps'][-1]:
        msg = 'The ICE cannot operate at an rps of '\
            + str(round(rps, 2)) + '.'
        raise ValueError(msg)
        
    # if the torque value is too high, give error message
    elif Q > PP['ICEMapTorque'][-1]:
        msg = 'The ICE cannot operate at a torque of '\
            + str(round(Q, 4)) + '.'
        raise ValueError(msg)
        
    elif Q == 0 or rps == 0:
        eff = 0

    # if Q and rps are fine, extrapolate
    else:
        eff = PP['ICEEffFun'](rps, Q)
    
    
    
    return float(eff)


def Sigmoid(t, floor, ceiling, start, end):
    """
    Implements a fully customizable sigmoid function. Used for transitions.

    Parameters
    ----------
    t : int or float
        x-axis value.
    floor : int or float
        Lower vertical limit.
    ceiling : float
        Higher Vertical limit.
    start : int or float
        Beginning of the positive slope.
    end : int or float
        End of the curve, sigmoid flattens here.


    Returns
    -------
    float
        y-value of the sigmoid.

    """

    
    a = ceiling - floor
    b = 8. / (end - start)
    # Using this offset the y value at t=0 is always 1.8% of the max value
    c = 4. + start * b
    
    return a / (1 + math.e ** (-b * t + c)) + floor




def PlantSizing(plant, rEM=0.6):
    
    PP = plant.copy()
    
    isa = at.Atmosphere()
    
    # aircraft design concept object creation requires a design-brief
    designbrief = {'rwyelevation_m': 0,  # altitudue of the runway
                   'groundrun_m': 50,  # maximumm allowed take-off distance
                   'climbrate_fpm': 3.5 * (60 / 0.3048),  # required climb rate that must be achieved
                   'climbspeed_kias': co.mps2kts(40),  # indidcated airspeed requirement for climb
                   'climbalt_m': 5000,  # altitude at which the climb rate must be achieved
                   'secclimbspd_kias': co.mps2kts(25),  # speed at which service ceiling is reached
                   'cruisespeed_ktas': co.mps2kts(40),  # cruise velocity
                   'cruisealt_m': 5000,  # altitude at which the cruise speed must be achieved
                   'cruisethrustfact': 1,  # ratio of cruise thrust to total thrust
                   'servceil_m': 5000,  # alt at which the max rate of climb drops to 100 ft/min
                   # dummy values to prevent errors, not needed since no turns are simulated
                   'turnspeed_ktas': 10,
                   'stloadfactor': 1.5
                   }
    
    # aircraft design concept object creation requires a design spec
    design = {'aspectratio': AC['AR'],
              'bpr': -3  # bypass ratio; -3 means no thrust correction (neglected for the aircraft)
              }
    
    
    
    # aircraft design concept object creation requires a performance estimate
    designperf = {'CDTO': GetCd(AC['maxPitch'], 0),  # take-off coefficient of drag
                  'CLTO': GetCl(AC['maxPitch']),  # take-off coefficient of lift
                  'CLmaxTO': GetCl(AC['maxPitch']),  # take-off maximum coefficient of lift
                  'CLmaxclean': GetCl(AC['maxPitch']),  # max lift coefficient in flight,
                  # with (non-existant) flaps retracted
                  'CDminclean': AC['parasiteDrag'],  # min, zero lift drag coefficient
                  'etaprop': {'take-off': 0.45, 'climb': 0.75, 'cruise': 0.85,
                              'turn': 0.85, 'servceil': 0.65},  # propeller efficiencies
                  }
    
    # An aircraft concept object can now be instantiated
    concept = ca.AircraftConcept(designbrief, design, designperf, isa)
    
    tow = PP['battMassList'][-1] + PP['fullTankMass'] + AC['emptyMass']\
        + AC['payloadMass'] + PP['EMMass'] + PP['ICEMass']
    
    wingloading = tow * AC['g'] / AC['S']
    
    power = concept.powerrequired(wingloading, tow, feasibleonly=False)
    
    # select largest power requirement, covnert from HP to W
    powerReq = max(power['take-off'], power['climb'], power['cruise'])
    
    powerReq = co.hp2kw(powerReq) * 1000
    
    # power is satisfied by ICE and EM
    ICEPowerReq = (1 - rEM) * powerReq
    EMPowerReq = rEM * powerReq
    
    
    # power available before sizing
    ICEPower = 2 * math.pi * PP['ICEMaprps'][-1] * PP['ICEMapTorque'][-1]
    EMPower = 2 * math.pi * PP['maxEMrps'] * PP['maxEMTorque']
    
    ICEFactor = ICEPowerReq / ICEPower
    EMFactor = EMPowerReq / EMPower
    
    # resize the torque scales to get adjusted power limits
    PP['ICEMapTorque'] *= ICEFactor
    PP['maxEMTorque'] *= EMFactor
    
    # remake the efficiency interpolation with new values
    PP['ICEEffFun'] = interp2d(PP['ICEMaprps'], PP['ICEMapTorque'],
                               PP['effMap'])

    PP['maxEMPower'] = 2 * math.pi * PP['maxEMrps'] * PP['maxEMTorque']
    
    # resize the mass
    PP['ICEMass'] *= ICEFactor
    PP['EMMass'] *= EMFactor
    
    # propeller sizing
    maxICETorque = PP['ICEMapTorque'][-1]
    maxICErps = PP['ICEMaprps'][-1]
    
    dFun = lambda D: maxICETorque - PropQFun(1, maxICErps, 0, D)
    
    PP['D'] = fsolve(dFun, 0.3)
    
    
    return PP


def PropPowerFun(v, rps, h, D=False):
    if not D:
        D = PP['D']
        
    if rps == 0:
        return 0
    
    else:
        J = v / (rps * D)
        
        PropCp = PP['CpFun'](J)
        
        # artificial fix if the J value passed is too far out of the interpolation range
        if PropCp < 0:
            PropCp = 0
        
        P = PropCp * Density(h) * rps ** 3 * D ** 5
        
        return P
    
    
def PropQFun(v, rps, h, D=False):
    if not D:
        D = PP['D']
        
    # prevent division by zero when calculating J
    if rps == 0:
        return 0
    
    else:
        J = v / (rps * D)
        
        PropCq = PP['CqFun'](J)
        
        # artificial fix if the J value passed is too far out of the interpolation range
        if PropCq < 0:
            PropCq = 0
            
        Q = PropCq * Density(h) * rps ** 2 * D ** 5
        
        return Q


def PropTFun(v, rps, h):
    # prevent division by zero when calculating J
    if rps == 0:
        return 0
    
    else:
        J = v / (rps * PP['D'])
        
        PropCt = PP['CtFun'](J)
        
        # artificial fix if the J value passed is too far out of the interpolation range
        if PropCt < 0:
            PropCt = 0
            
        T = PropCt * Density(h) * rps ** 2 * PP['D'] ** 4
        
        return T


AC, PP = GetParams('aircraft_params.txt')
PP = PlantSizing(PP)
