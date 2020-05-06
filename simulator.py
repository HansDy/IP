""" Main file of the simulation package """

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint, cumtrapz, ode
from scipy.optimize import fsolve, brentq

from support_funs import GetCd, GetCl, Sigmoid, PropTFun, PropPowerFun, PropQFun, ICEEff, AC, \
    PlantSizing
from get_params import Density, GetParams
 
 
def dcos(x):
    return math.cos(math.radians(x))


def dsin(x):
    return math.sin(math.radians(x))


def dtan(x):
    return dsin(x) / dcos(x)


def datan(x1, x2):
    return math.degrees(math.atan2(x1, x2))


class simulate(object):
    def __init__(self, rng, plant=None, runSizing=True):
        self.plotMarkers = np.array([]);   self.rng = rng * 1000
        self.runtime = 0.;                 self.integrator = 'odeint'
        self.steps = 0;                    self.time = 0.
        self.payloadDropped = False
        
        if not plant:
            _, self.PP = GetParams()
        else:
            self.PP = plant.copy()
        
        # size the powerplant, get updated dictionary to store in object
        if runSizing:
            self.PP = PlantSizing(self.PP)
        
        self.battChoice = int(self.PP['battChoice'])
        
        self.battCap = self.PP['battCapList'][self.battChoice]
        self.battSE = self.PP['battSEList'][self.battChoice]
        self.battMass = self.PP['battMassList'][self.battChoice]
        
        # zero fuel mass - everything but fuel is included here
        self.zfm = self.battMass + AC['emptyMass'] + AC['payloadMass']
        
        self.totalMass = AC['emptyMass'] + AC['payloadMass'] + self.PP['fullTankMass']\
            + self.battMass + self.PP['EMMass'] + self.PP['ICEMass']

        # contains x1, x2, v1, v2 in four columns
        self.solution = False

        # vars for sigmoid schedules
        self.TFloor = 0.;                  self.pitchFloor = 0.
        self.TCeiling = 0.;                self.pitchCeiling = 0.
        self.TStart = 0.;                  self.pitchStart = 0.
        self.TEnd = 0.;                    self.pitchEnd = 0.
        self.TShape = 'const';             self.pitchShape = 'const'

        # total amount of data, total time steps returned from ODE solver
        self.n = 0

        # initial values for ode solver
        self.y0 = np.zeros(4)

        """ data arrays """

        # angles
        self.alpha = np.zeros(10**3)
        self.pitch = np.zeros(10**3)
        self.gamma = np.zeros(10**3)
        
        # kinematics
        self.x1 = np.zeros(10**3);      self.x2 = np.zeros(10**3)
        self.v1 = np.zeros(10**3);      self.v2 = np.zeros(10**3)
        self.a1 = np.zeros(10**3);      self.a2 = np.zeros(10**3)
        
        
        # time that corresponds with each index
        self.timeArray = np.zeros(10**3)
        
        # thrust
        self.thrust = np.zeros(10**3)
        self.EMThrust = np.zeros(10**3)
        self.ICEThrust = np.zeros(10**3)

        # energy related
        self.totalE = np.zeros(10**3);      self.fuelMass = np.zeros(10**3)
        self.fuelE = np.zeros(10**3);       self.SOC = np.zeros(10**3)
        self.battE = np.zeros(10**3)

        # power related
        self.power = np.zeros(10**3)  # total power coming out of the prop
        self.fuelP = np.zeros(10**3)  # power consumed by ICE
        self.battP = np.zeros(10**3)  # power consumed by EM
        self.ICEShaftP = np.zeros(10**3)
        self.EMShaftP = np.zeros(10**3)
        self.ICEPropP = np.zeros(10**3)  # total power produced by the ICE connected propeller
        self.EMPropP = np.zeros(10**3)  # total power produced by the EM connected propeller
        
        self.ICEEff = np.zeros(10**3)  # ICE brake efficiency 
        
        ### propeller related ###
        # -- EM Propeller -- #               # -- ICE Propeller -- #
        self.EMrps = np.zeros(10**3);        self.ICErps = np.zeros(10**3)
        self.EMJ = np.zeros(10**3);          self.ICEJ = np.zeros(10**3)
        self.EMPropEff = np.zeros(10**3);    self.ICEPropEff = np.zeros(10**3)
        
        self.EMTotEff = np.zeros(10**3)
        self.ICETotEff = np.zeros(10**3)
        self.rEM = np.zeros(10**3)  # ratio of output power provided by EM
        
        # 0 - off, 1 - ICE only, 2 - ICE ideal and EM, 3 - EM max and ICE, 4 - excess
        self.RBOut = np.zeros(10**3)
        self.massCost = np.zeros(10**3)


    def Equations(self, t, y):
        """Returns the equations of motion to be integrated by the
        ODE integrator."""
        # y must be an array with elements x1 x2 v1 and v2 which is passed by the
        # ODE solver

        x1, x2, v1, v2 = y

        dydt = [0, 0, 0, 0]

        dydt[0] = v1
        
        dydt[1] = v2
        
        dydt[2] = (GetCoeffA(self.GetAlpha(t, v1, v2), self.GetGamma(v1, v2), x2)
                   * (v1**2 + v2**2) + self.TSchedule(t) * dcos(self.PitchSchedule(t)))\
            / self.totalMass
            
        dydt[3] = (GetCoeffB(self.GetAlpha(t, v1, v2), self.GetGamma(v1, v2), x2)
                   * (v1**2 + v2**2) + self.TSchedule(t) * dsin(self.PitchSchedule(t)))\
            / self.totalMass - AC['g']

        # reaction force implementation
        if GetCoeffB(self.GetAlpha(t, dydt[0], dydt[1]), self.GetGamma(v1, v2), x2) * (dydt[0]**2) \
                < self.totalMass * AC['g'] and x2 == 0:
            dydt[3] = 0

        return dydt


    def GetGamma(self, v1, v2):
        return math.degrees(math.atan2(v2, v1))


    def GetAlpha(self, t, v1, v2):
        return self.PitchSchedule(t) - self.GetGamma(v1, v2)


    def TSchedule(self, t):
        """Returns the required thrust output signal in Newtons."""
        if self.TShape == 'const':
            return self.TCeiling

        elif self.TShape == 'sigmoid':
            return Sigmoid(t, self.TFloor, self.TCeiling,
                           self.TStart, self.TEnd)


    def PitchSchedule(self, t):
        if self.pitchShape == 'const':
            return self.pitchCeiling

        elif self.pitchShape == 'sigmoid':
            return Sigmoid(t, self.pitchFloor, self.pitchCeiling,
                           self.pitchStart, self.pitchEnd)

        else:
            raise ValueError('pitchShape variable named incorrectly')


    def PDemand(self):
        """Power that must come out of the powerplant overall (returns in kW)"""
        return self.V() * self.thrust[self.n]


    """f(v1, v2)"""
    def V(self):
        """get the absolute value of the velocity in m/s"""
        V = math.sqrt(self.v1[self.n]**2 + self.v2[self.n]**2)
        return V


    def RunInterval(self):
        """Runs the simulation, updates all the data. List of tasks completed:
            1 - runs the ode solver (integrate.ode or integrate.odeint) and obtains
                  the position and velocity values
            
            2 - ensures the data arrays are long enough to store the incoming data
            
            3 - Set plot markers where the interval was called
            
            4 - run loop through ode solver steps taken in two to process data
            4.1 - keep track of the size of the time-steps taken by storing the time
                    associated with each index of the data arrays
            4.2 - extract all the velocity and position data from the solution array
                    obtained from numerical ode solver
            4.3 - store the control inputs as they were in time (pitch and thrust),
                    as well as the other relevant angles (gamma, alpha)
            4.4 - call the RB controller to allocate the thrust demand between the ICE and
                    EM. This will set the ICE and EM thrust and speed
            4.5 - calculate and store values for the power consumed by the ICE, EM and in total
            4.6 - drop the payload if halfway through the flight

            5 - calculate and store the horizontal and vertical acceleration
                    values (resultant acceleration) by differentiating velocity data
            
            5 - keep track of the times at which this RunInterval() method was
                    called to keep track of separate interval calls

            6 - Set initial values for the next interval by taking the last values from the
                    current solution array. Update the time attribute

            7 - Ensures the default setting for the thrust and pitch schedules is
                  constant. This makes quickly changing the control inputs from
                 the command line quicker.
        """

        
        # check the time is running forward
        if self.runtime < 0:
            raise ValueError('RunInterval() is receiving a negative runtime'
                             + ' of' + str(round(self.runtime, 2)))
        elif self.runtime == 0:
            raise ValueError('RunInterval() is receiving a runtime of zero')
        
        self.steps = int(self.steps)
        
        t = np.linspace(0, self.runtime, self.steps + 1)

        # make sure the timesteps is 2 or greater to allow the np.gradient to run
        if self.steps < 2:
            raise ValueError('RunInterval has been called by a function with'
                             'less than two steps per Interval')

        """ 1. get x1, x2, v1 and v2 by solving the equations of
            motion numerically """
        
        if self.integrator == 'ode':
            # create empty 2d array to store the solution
            self.solution = np.zeros((self.steps, 4))
    
            # create ode object to compute values
            eq = ode(self.Equations).set_initial_value(self.y0).set_integrator('vode')
    
            # define time-step
            dt = self.runtime / self.steps
    
            for i in range(self.steps):
                self.solution[i, :] = eq.integrate(eq.t + dt)
                
        elif self.integrator == 'odeint':
            self.solution = odeint(self.Equations, self.y0, t, tfirst=True)

        # print('Solution length: ', len(self.solution[:, 0]), '\n')

        """ 2. make sure that the list of zeros is long enough,
            append more if not. Numpy appending is slow, so this way it is kept
            to  minimum """
        if len(self.x1) <= self.n + len(self.solution[:, 0]) or self.steps > 10**3:
            self.x1 = np.append(self.x1, np.zeros(10**3))
            self.x2 = np.append(self.x2, np.zeros(10**3))
            self.v1 = np.append(self.v1, np.zeros(10**3))
            self.v2 = np.append(self.v2, np.zeros(10**3))
            self.a1 = np.append(self.a1, np.zeros(10**3))
            self.a2 = np.append(self.a2, np.zeros(10**3))
            
            self.timeArray = np.append(self.timeArray, np.zeros(10**3))
            
            self.alpha = np.append(self.alpha, np.zeros(10**3))
            self.pitch = np.append(self.pitch, np.zeros(10**3))
            self.gamma = np.append(self.gamma, np.zeros(10**3))
            
            self.thrust = np.append(self.thrust, np.zeros(10**3))
            self.ICEThrust = np.append(self.ICEThrust, np.zeros(10**3))
            self.EMThrust = np.append(self.EMThrust, np.zeros(10**3))

            self.power = np.append(self.power, np.zeros(10**3))
            self.fuelP = np.append(self.fuelP, np.zeros(10**3))
            self.battP = np.append(self.battP, np.zeros(10**3))
            self.ICEShaftP = np.append(self.ICEShaftP, np.zeros(10**3))
            self.EMShaftP = np.append(self.EMShaftP, np.zeros(10**3))
            self.ICEPropP = np.append(self.ICEPropP, np.zeros(10**3))
            self.EMPropP = np.append(self.EMPropP, np.zeros(10**3))

            self.SOC = np.append(self.SOC, np.zeros(10**3))
            self.fuelMass = np.append(self.fuelMass, np.zeros(10**3))
            self.fuelE = np.append(self.fuelE, np.zeros(10**3))
            self.battE = np.append(self.battE, np.zeros(10**3))
            self.totalE = np.append(self.totalE, np.zeros(10**3))
            
            self.EMrps = np.append(self.EMrps, np.zeros(10**3))
            self.ICErps = np.append(self.ICErps, np.zeros(10**3))
            self.ICEJ = np.append(self.ICEJ, np.zeros(10**3))
            self.EMJ = np.append(self.EMJ, np.zeros(10**3))
            self.EMPropEff = np.append(self.EMPropEff, np.zeros(10**3))
            self.ICEPropEff = np.append(self.ICEPropEff, np.zeros(10**3))
            
            self.EMTotEff = np.append(self.EMTotEff, np.zeros(10**3))
            self.ICETotEff = np.append(self.ICETotEff, np.zeros(10**3))
            self.rEM = np.append(self.rEM, np.zeros(10**3))
            self.RBOut = np.append(self.RBOut, np.zeros(10**3))
            
            self.massCost = np.append(self.massCost, np.zeros(10**3))
            self.ICEEff = np.append(self.ICEEff, np.zeros(10**3))
            
        """ 3. fill array for markers where odeint restarted"""
        self.plotMarkers = np.append(self.plotMarkers, int(self.n))


        """ 4. Loop to fill the data lists x1, x2, v1 etc."""
        if self.n == 0:
            m = len(self.solution[:, 0])
        else:
            m = len(self.solution[:, 0]) - 1
        for i in range(m):
            if self.n - self.steps > 0:
                i += 1
            
            self.n += 1
            
            """ 4.1 Keep track of time """
            self.timeArray[self.n] = self.timeArray[self.n - 1] + self.runtime / self.steps

            """ 4.2 Set displacement and velocity values"""
            self.x1[self.n] = self.solution[i, 0]
            self.x2[self.n] = self.solution[i, 1]
            self.v1[self.n] = self.solution[i, 2]
            self.v2[self.n] = self.solution[i, 3]

            """ 4.3 fill thrust, alpha and pitch arrays"""
            self.thrust[self.n] = self.TSchedule(t[i])
            self.alpha[self.n] = self.GetAlpha(t[i], self.v1[self.n], self.v2[self.n])
            self.pitch[self.n] = self.PitchSchedule(t[i])
            self.gamma[self.n] = self.GetGamma(self.v1[self.n], self.v2[self.n])

            """ 4.4 RB controller to allocate the thrust """
            self.AllocateThrust()
            
            """ 4.5 Find power related values """
            
            # RBOut == 4 is a flag set by the RB to indicate excess thrust was demanded
            if self.RBOut[self.n] == 4:
                # Set prohibitively high power value to disincentivize optimizer
                self.EMShaftP[self.n] = 10e5
                self.ICEShaftP[self.n] = 10e5
            
            else:
                self.EMShaftP[self.n] = PropPowerFun(v=self.V(), rps=self.EMrps[self.n],
                                                     h=self.x2[self.n])
                self.ICEShaftP[self.n] = PropPowerFun(v=self.V(), rps=self.ICErps[self.n],
                                                      h=self.x2[self.n])
            
            # propeller output powers
            self.ICEPropP[self.n] = self.ICEThrust[self.n] * self.V()
            self.EMPropP[self.n] = self.EMThrust[self.n] * self.V()
            self.power[self.n] = self.PDemand()
                        
            
            # if the propeller is not spinning, there cannot be a J value (division by zero)
            if self.ICErps[self.n] == 0:
                self.ICEJ[self.n] = np.nan
                ICETorque = 0

            else:
                self.ICEJ[self.n] = self.V() / (self.PP['D'] * self.ICErps[self.n])
                # needed later because ICEEff = f(Torque)
                ICETorque = PropQFun(self.V(), self.ICErps[self.n], self.x2[self.n])
            
            # if the propeller is not spinning, there cannot be a J value (division by zero)
            if self.EMrps[self.n] == 0:
                self.EMJ[self.n] == np.nan
            
            else:
                self.EMJ[self.n] = self.V() / (self.PP['D'] * self.EMrps[self.n])
                
            # prevent a call to the ICEEff function if excess torque is demanded from ICE
            if self.RBOut[self.n] == 4:
                self.ICEEff[self.n] = 0.01
            else:
                self.ICEEff[self.n] = ICEEff(self.PP, self.ICErps[self.n], ICETorque)
                
            # find power drawn from battery and fuel tank
            self.battP[self.n] = self.EMShaftP[self.n] / self.PP['EMEff']
            if self.ICEEff[self.n] == 0:  # the ICE eff. is zero when the prop is not spinning
                self.fuelP[self.n] = 0
            else:
                self.fuelP[self.n] = self.ICEShaftP[self.n] / self.ICEEff[self.n]

            """ 4.6 drop payload halfway through the flight """
            if self.x1[self.n] >= self.rng / 2 and not self.payloadDropped:
                self.DropPayload()
                self.payloadDropped = self.x1[self.n]
                # print('Payload Dropped at distance =', round(self.x1[self.n]/1000, 2), 'km',
                #      'and time =', self.timeArray[self.n], 's')


        s = self.n - m
        """ 5. find the accelerations by differentiation"""
        # direct differentiation from the solution array
        self.a1[s:s + m] = np.gradient(self.solution[:m, 2])
        self.a2[s:s + m] = np.gradient(self.solution[:m, 3])


        """ 6. update number of simulation steps taken, the total time that has
            passed and the initial conditions for the ODE solver """
        self.time = self.timeArray[self.n]
        self.oldy0 = self.y0
        self.y0[0] = self.solution[-1, 0]
        self.y0[1] = self.solution[-1, 1]
        self.y0[2] = self.solution[-1, 2]
        self.y0[3] = self.solution[-1, 3]
        
        # print('solution: ', self.solution[-1,:],'\n')
        
        """ 7. Ensure the pitch and thrust input signal types are reset """
        self.TShape = 'const'
        self.pitchShape = 'const'

    def CalculateE(self):
        """
        Cumulatively integrate simulator power arrays to find energy values and mass cost

        Returns
        -------
        None.

        """
        e = self.n
        self.fuelE = cumtrapz(self.fuelP[0:e], self.timeArray[0:e],
                              initial=self.fuelE[0])
    
        self.battE = cumtrapz(self.battP[0:e], self.timeArray[0:e],
                              initial=self.battE[0])
        
        self.totalE = self.battE + self.fuelE

        self.SOC = (1 - self.battE / self.battCap)
        self.fuelMass = self.PP['fullTankMass'] - self.fuelE / self.PP['LCV']
        
        self.EMTotEff = self.EMPropP / (self.battP + 1e-5)
        self.ICETotEff = self.ICEPropP / (self.fuelP + 1e-5)
        self.rEM = self.EMPropP / (self.EMPropP + self.ICEPropP + 1e-5)
        
        self.rEM = self.EMThrust / (self.thrust + 1e-5)

        self.battMassCost = self.battE / self.battSE
        self.fuelMassCost = self.PP['fullTankMass'] - self.fuelMass
        self.massCost = self.battMassCost + self.fuelMassCost


    def AllocateThrust(self):
        """
        RB controller. Determines the ICE and EM speeds and thrust by applying a set
        of rules around the ICE speed which produces the highest efficiency. If excess thrust
        is demanded, the RBOut attribute array is flagged with a 4 at the relevant n value

        Returns
        -------
        None.

        """
        v = self.V()
        h = self.x2[self.n]
        thrustRequest = self.thrust[self.n]
        
        idealICEThrust = PropTFun(v, self.PP['idealICErps'], h)
        
        maxICEThrust = PropTFun(v, self.PP['ICEMaprps'][-1], h)
    
        maxEMThrust = PropTFun(v, self.PP['maxEMrps'], h)
        
        # print('Requested Thrust =', thrustRequest)
        # print('Max EM thrust available=', maxEMThrust[0])
        # print('Max ICE thrust available=', maxICEThrust[0])
        
        # use a guess which would produce the ideal J for the prop eff
        rpsGuess = v / (self.PP['D'] * self.PP['optJ'])
        if rpsGuess < 10:
            rpsGuess = 10
        
        # warn if thrust cannot be given, then set the rpm to the necessary value
        # this will be filtered out later and converted to a disproportionately high energy cost
        if thrustRequest > maxICEThrust + maxEMThrust:
            print('\nWARNING: THE THRUST REQUESTED IS NOT AVAILABLE.',
                  thrustRequest, 'N were requested but only',
                  maxICEThrust + maxEMThrust, 'are available')
            print('Current velocity:', self.V(), '\n')
            
            excess = thrustRequest - maxICEThrust - maxEMThrust
            
            ICEThrust = maxICEThrust + excess / 2
            EMThrust = maxEMThrust + excess / 2
            try:
                ICErps = brentq(lambda rps: PropTFun(v, rps, h) - ICEThrust,
                                self.PP['ICEMaprps'][0], self.PP['ICEMaprps'][-1] * 2)
            except ValueError:
                ICErps = self.PP['ICEMaprps'][-1]
            
            try:
                EMrps = brentq(lambda rps: PropTFun(v, rps, h) - EMThrust,
                               0, self.PP['maxEMrps'] * 2)
            except ValueError:
                EMrps = self.PP['maxEMrps']
            
            # flag to dramatically increase thrust demand
            self.RBOut[self.n] = 4
        
        # use max EM thrust and lowest possible ICE thrust
        elif thrustRequest > idealICEThrust + maxEMThrust:
            EMThrust = maxEMThrust
            ICEThrust = thrustRequest - EMThrust
            
            EMrps = self.PP['maxEMrps']

            ICErps = brentq(lambda rps: PropTFun(v, rps, h) - ICEThrust,
                            self.PP['ICEMaprps'][0], self.PP['ICEMaprps'][-1])
            self.RBOut[self.n] = 3
    
        # use ideal ICE thrust and the rest is EM power
        elif thrustRequest > idealICEThrust:
            ICEThrust = idealICEThrust
            EMThrust = thrustRequest - ICEThrust

            ICErps = self.PP['idealICErps']
            EMrps = brentq(lambda rps: PropTFun(v, rps, h) - EMThrust,
                           0, self.PP['maxEMrps'])

            self.RBOut[self.n] = 2
    
        # use only ICE
        elif thrustRequest <= idealICEThrust and thrustRequest > 0:
            ICEThrust = thrustRequest
            EMThrust = 0
            
            EMrps = 0
            
            # if the thrust is tiny, set it to zero
            if PropTFun(v, 1, h) - ICEThrust > 0:
                ICEThrust = 0
                ICErps = 0
                
                self.RBOut[self.n] = 0
                
            else:
                ICErps = brentq(lambda rps: PropTFun(v, rps, h) - ICEThrust,
                                1, self.PP['ICEMaprps'][-1])
            
                self.RBOut[self.n] = 1
        
        # plant is off
        elif thrustRequest == 0:
            ICEThrust = 0
            EMThrust = 0
            
            EMrps = 0
            ICErps = 0
            self.RBOut[self.n] = 0
            
        self.EMrps[self.n] = EMrps
        self.ICErps[self.n] = ICErps
        self.EMThrust[self.n] = EMThrust
        self.ICEThrust[self.n] = ICEThrust
        
        
    def IdealTRange(self, *args):
        """ Returns the total (EM+ICE)thrust range at which the powerplant
        can run the ICE at ideal rps"""
        
        if len(args) == 0:
            v = self.V()
            h = self.x2[self.n]
            
        else:
            v = args[0]
            h = args[1]
        
        lower = PropTFun(v, self.PP['idealICErps'], h)[0]
    
        upper = (PropTFun(v, self.PP['maxEMrps'], h) + lower)[0]
            
        return lower, upper
    
    def DropPayload(self):
        """
        Drops the payload by reducing the aircaft mass by the payload mass.

        Returns
        -------
        None.

        """
        self.zfm -= AC['payloadMass']
        self.totalMass -= AC['payloadMass']
        # print('Payload dropped at x1 = ', self.x1[self.n])


def GetCoeffA(alpha, gamma, h):
    """
    Returns a coefficient for the (horizontal) equations of motion.Written out here to
    keep the EOMs shorter and neater. Should only be called the function which implements
    the EOMs for scipy.integrate.ode(int).

    Parameters
    ----------
    alpha : int or float
        Angle of attack.
    gamma : int or float
        Flight path angle.
    h : int or float
        Altitude.

    Raises
    ------
    ValueError
        When args of wrong type are passed

    Returns
    -------
    float
        Value of the coefficient.
    """

    return -0.5 * Density(h) * AC['S'] \
        * (GetCd(alpha, h) * dcos(gamma) + GetCl(alpha) * dsin(gamma))


def GetCoeffB(alpha, gamma, h):
    """
    Returns a coefficient for the (vertical) equations of motion.Written out here to
    keep the EOMs shorter and neater. Should only be called the function which implements
    the EOMs for scipy.integrate.ode(int).

    Parameters
    ----------
    alpha : int or float
        Angle of attack.
    gamma : int or float
        Flight path angle.
    h : int or float
        Altitude.

    Raises
    ------
    ValueError
        When args of wrong type are passed

    Returns
    -------
    float
        Value of the coefficient.

    """

    return (0.5 * Density(h) * AC['S'] * (GetCl(alpha) * dcos(gamma)
            - GetCd(alpha, h) * dsin(gamma)))
