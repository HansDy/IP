from scipy.optimize import fsolve
import math


from support_funs import VLO, ZeroPitchCruiseV, GetCl, GetCd, ClCd, Sigmoid, PP, AC, PropTFun
from get_params import Density


def dcos(x):
    return math.cos(math.radians(x))


def dsin(x):
    return math.sin(math.radians(x))


def dtan(x):
    return dsin(x) / dcos(x)


def datan(x1, x2):
    return math.degrees(math.atan2(x1, x2))


def LevelFlight(s, h, vReq):
    """Equations of motion for alpha = pitch, v1 = vReq and
    v2 = 0 = a2 = a1 = 0 to return the correct pitch and thrust to 
    obtain the required cruise velocity
    

    Parameters
    ----------
    s : simulate OBJECT
    h : int or float
        Altitude.
    vReq : int or float
        Required cruise velocity.

    Returns
    -------
    T : float
        Needed thrust signal.
    pitch : float
        Needed pitch signal.

    """
    
    # Determine on which side of the discontinuity to start the numerical root finder
    levelFlightV = ZeroPitchCruiseV(s)
    
    if levelFlightV > vReq:
        initialGuess = 0.1
    else:
        initialGuess = -0.1

    W = s.totalMass * AC['g']
    
    pitch = fsolve(lambda p: -0.5 * Density(h) * AC['S'] * vReq ** 2 * \
                   (GetCl(p) + dtan(p) * GetCd(p, h)) + W,
                   initialGuess)
    T = 0.5 * Density(h) * AC['S'] * vReq ** 2 * GetCd(pitch, h) / dcos(pitch)

    return T, pitch



def AnalyticalFindTP(s, h, v1, v2):
    """ Finds solution for the given parameters, solving the rearranged EOM
    analytically. Returns T and pitch needed """

    # find the flight path angle
    gamma = math.degrees(math.atan2(v2, v1))
    
    # find the total velocity
    V = math.sqrt(v1 ** 2 + v2 ** 2)
    
    # if v1 is above the zero-pitch flight velocity this should be negative
    if ZeroPitchCruiseV(s, 0) < v2:
        initialGuess = -1
        
    else:
        initialGuess = 1

        
    f = lambda p: 0.5 * Density(h) * AC['S'] * V ** 2 *\
                   ( GetCl(p-gamma) * (dsin(gamma)/dcos(p) + dcos(gamma)/dsin(p)) +\
                   GetCd(p-gamma, h) * (dcos(gamma)/dcos(p) - dsin(gamma)/dsin(p)) )\
                   - s.totalMass * AC['g'] / dsin(p)
    
    pitch = fsolve(f, initialGuess)
    
    # fsolve fails sometimes, can't figure out why. Initial guess changing doesnt seem to work
    
    if abs(f(pitch)) > 0.1:
        pitch = fsolve(f, -initialGuess)
        if abs(f(pitch)) > 0.1:
            print('AnalyticalFindTP() could not find the zero on the second attempt.', 
                  pitch, v1, v2, f(pitch))
            raise ValueError('fsolve failed \n')

        
    T = 0.5 * Density(h) * AC['S'] * V ** 2 / dcos(pitch) * \
        (GetCl(pitch-gamma) * dsin(gamma) + GetCd(pitch-gamma, h) * dcos(gamma))
        
    T2 = 0.5 * Density(h) * AC['S'] * V ** 2 / dsin(pitch) *\
        (GetCd(pitch-gamma, h) * dsin(gamma) - GetCl(pitch-gamma) * dcos(gamma)) + \
        s.totalMass * AC['g'] / dsin(pitch)
        
    if abs(T - T2) > 1e-5:
        print('Error calculating T in AnalyticalFindTP()')
        print(T, T2, pitch, v1, v2)
        
        
        
    return T, pitch



def TakeOff(s, minimizeTOD = False, fullOutput=True):

    # liftoff velocity at 0 angle of attack
    vlo = VLO(s)
    
    # make sure the plane can actually take off before proceeding
    if s.time != 0 or s.n != 0 or\
     s.x2[s.n] != 0:
         raise ValueError('The aircraft is not at take-off conditions')
         
    if fullOutput:
        print('\n----------')
        print('-- Take Off --\n')
        print('Expected liftoff v: ', round(VLO(s, 8), 2), ' m/s')
        print('\n')
        
    # take off at maximum thrust
    if minimizeTOD:
        s.runtime = 1
        s.steps = 2
        
        h = 0
        
        # add maximum thrusts
        T = PropTFun(vlo, s.PP['ICEMaprps'][-1], 0) + PropTFun(vlo, s.PP['maxEMrps'], 0)
    
        s.TShape = 'const';             s.pitchShape = 'const'

        while h <= 0:
            
            if s.V() > vlo:
                print('Readjusting liftoff velocity')
                T = PropTFun(s.V() * 1.2, PP['ICEMapShaftrps[-1]'], 0) \
                    + PropTFun(s.V() * 1.2, PP['maxEMrps'], 0)
                s.TShape = 'const'
            
            s.TShape = 'const';             s.pitchShape = 'const'
            s.TCeiling = Sigmoid(s.time, 0, T, 0, 10)
            s.pitchCeiling = Sigmoid(s.time, 0, AC['maxPitch'], 0, 5)

            h = s.x2[s.n]
            
            s.RunInterval()
    
    # run at max EM thrust plus ideal ICE thrust
    else:
        
        # set thrust to 90% of the upper limit to run the ICE at ideal rps
        T = s.IdealTRange(vlo, 0)[1] * 0.9

        s.runtime = 1
        s.steps = 2
        
        s.TShape = 'const'
        s.pitchShape = 'const'
        
        h = s.x2[s.n]
        
        while h <= 0:
            
            if s.V() > vlo:
                print('Readjusting liftoff velocity')
                T = PropTFun(s.V() * 1.2, s.PP['ICEMaprps'][-1], 0)\
                + PropTFun(s.V() * 1.2, s.PP['maxEMrps'], 0)
                s.TShape = 'const'
                s.TCeiling = T
                
            s.TShape = 'const';             s.pitchShape = 'const'
            s.TCeiling = Sigmoid(s.time, 0, T, 0, 10)
            s.pitchCeiling = Sigmoid(s.time, 0, AC['maxPitch'], 0, 5)


            h = s.x2[s.n]
            
            s.RunInterval()

    
    if fullOutput:
        print('Lift off achieved in ', s.time, 's, ',
              '\n    x1=', round(s.x1[s.n], 2), 'm | v1 =', round(s.v1[s.n], 2), 'm/s',
              '\n    x2=', round(s.x2[s.n], 2), 'm | v2 =', round(s.v2[s.n], 2), 'm/s',
              '\n    thrust =', round(s.thrust[s.n - 1], 2), 'N | pitch =',
              round(s.pitch[s.n - 1], 2), 'degrees')
        print('--------')


def Climb(s, v1=30, desiredCR=1, desiredHeight=1000, fullOutput=True):
    """
    Sets the aircraft on a climb up to a given height and at a given climb rate.

    Parameters
    ----------
    s : simulate OBJECT
    v1 : float, optional
        Horizontal velocity component to be achieved. The default is 30.
    desiredCR : float, optional
        Vertical velocity component to be achieved. The default is 1.
    desiredHeight : float, optional
        The height at which the climb should end. The default is 1000.
    fullOutput : bool, optional
        Determine whether to mute the print statements. The default is True.


    Returns
    -------
    None.

    """
    
    if v1 < 0:
        msg = 'Climb function was passed negative value for v1. v1 = ' + str(round(v1, 2)) + 'm/s'
        raise ValueError(msg)
        
    if desiredCR < 0:
        msg = 'Climb function was passed negative value for CR. CR = '\
            + str(round(desiredCR, 2)) + 'm/s'
        raise ValueError(msg)
        
    if desiredHeight < 0:
        msg = 'Climb function was passed negative value for desired Height. h = '\
            + str(round(desiredHeight, 2)) + 'm'
        raise ValueError(msg)
        
    if desiredHeight - s.x2[s.n] < 0:
        msg = 'The climb function cannot climb below the current height. desired height h = '\
            + str(round(desiredHeight, 2)) + 'm, current height = ' + str(round(s.x2[s.n]))\
            + 'm. But the desired height must be greater than the current height.'
        raise ValueError(msg)

    
    # save for later
    initialHeight = s.x2[s.n]
    initialTime = s.time
    v2 = desiredCR
    
    if s.n == 0:
        raise ValueError('Climb cannot start before take-off')
    else:
        h = s.x2[s.n]


    # set the control input values
    T, pitch = AnalyticalFindTP(s, h, v1, v2)

    # convert to int
    T = T[0]
    pitch = pitch[0]


    # print info on the impending climb (or descent)
    if fullOutput:
        print('\n----------')
        print('-- CLIMB --')
        print('Starting climb at t=', s.time,
              '\n    x1=', round(s.x1[s.n], 2), 'm | v1 =', round(s.v1[s.n], 2), 'm/s',
              '\n    x2=', round(s.x2[s.n], 2), 'm | v2 =', round(s.v2[s.n], 2), 'm/s',
              '\n    desiredCR =', desiredCR, 'm/s'
              '\n    desired final alt. =', round(desiredHeight / 1000, 3), 'km',
              '\n    thrust setting =', round(T, 2), 'N',
              '\n    pitch setting =', round(pitch, 2), 'degrees \n')


    ########## START ##########
    # adjust the time the aircaft turns based on the size of the turn
    turnDuration = 5 * abs(AC['maxTurnRate'] / (pitch - s.pitch[s.n]))

    # set the sigmoid transitions for control inputs
    s.TShape = 'sigmoid';             s.pitchShape = 'sigmoid'
    s.TFloor = s.thrust[s.n];         s.pitchFloor = s.pitch[s.n]
    s.TCeiling = T;                   s.pitchCeiling = pitch
    s.TStart = 0;                     s.pitchStart = 0
    s.TEnd = 10;                      s.pitchEnd = turnDuration

    # run for a bit to let the CR settle into the right value
    s.runtime = 15
    s.steps = 15
    s.RunInterval()
    ########## END ##########

    # assume the settled region is reached, record height and time
    settledTime = s.time
    settledHeight = s.x2[s.n]

    # as the air density changes with altitude, it is necessary to restart the interval
    # at least every 800m:
    nIntervals = math.floor((desiredHeight - s.x2[s.n]) / 800)

    # catch cases where the desired climb altitude is less than 800m
    if nIntervals < 1:
        nIntervals = 1

    ########## START ##########
    # runtime for the simulator, should start new interval roughly every 500 m of gained alt
    intervalLength = (desiredHeight - s.x2[s.n]) / abs(desiredCR) / (nIntervals + 1)

    s.runtime = intervalLength
    s.steps = 15

    # climb until the desiredHeight is almost reached
    for i in range(nIntervals):
        h = s.x2[s.n]

        # readjust control inputs
        T, pitch = AnalyticalFindTP(s, h, v1, v2)

        s.TShape = 'sigmoid';             s.pitchShape = 'sigmoid'
        s.TFloor = s.thrust[s.n];         s.pitchFloor = s.pitch[s.n]
        s.TCeiling = T;                   s.pitchCeiling = pitch
        s.TStart = 0;                     s.pitchStart = 0
        s.TEnd = 3;                       s.pitchEnd = 3

        s.RunInterval()
    ########## END ##########

    ########## START ##########
    # get more precise value for the climb rate that is actually
    # reached by the aircraft
    actualCR = s.v2[s.n]

    s.runtime = (desiredHeight - s.x2[s.n]) / actualCR
    s.steps = 10

    h = s.x2[s.n]

    # readjust control inputs
    T, pitch = AnalyticalFindTP(s, h, v1, v2)

    s.TShape = 'sigmoid';             s.pitchShape = 'sigmoid'
    s.TFloor = s.thrust[s.n];         s.pitchFloor = s.pitch[s.n]
    s.TCeiling = T;                   s.pitchCeiling = pitch
    s.TStart = 0;                     s.pitchStart = 0
    s.TEnd = 3;                       s.pitchEnd = 3

    s.RunInterval()
    ########## END ##########

    # values for printing to console
    climbDistance = s.x2[s.n] - initialHeight
    climbTime = s.time - initialTime
    avgCR = (s.x2[s.n] - settledHeight) / (s.time - settledTime)

    # print warning if the operator did not achieve the desired climb (within a
    # margin of 10%)
    if abs((avgCR - desiredCR) / desiredCR) > 0.1:
        print('\nWARNING: THE OPERATOR DID NOT ACHIEVE THE CORRECT CLIMB RATE.',
              '\nTHE DESIRED CR IS ', round(desiredCR, 2), 'm/s BUT THE ACHIEVED CR',
              'IS', round(avgCR, 2), 'm/s. (Alt=', round(s.x2[s.n] / 1000, 3),
              'km, v1=', round(v1, 3), 'm/s)\n')

    # print warning if the operator did not achieve the desired height (within a
    # margin of 10%)
    if abs((s.x2[s.n] - desiredHeight) / desiredHeight) > 0.1:
        print('\nWARNING: THE OPERATOR DID NOT ACHIEVE THE CORRECT HEIGHT AFTER',
              'THE CLIMB. THE DESIRED HEIGHT IS ', desiredHeight,
              'm BUT THE ACHIEVED HEIGHT IS', s.x2[s.n], 'm (v1=', round(v1, 3),
              'm/s, v2=', round(desiredCR, 3), 'm/s)\n')

    if fullOutput:
        print('Finished climb at t=', s.time,
              '\n    x1=', round(s.x1[s.n], 2), 'm | v1 =', round(s.v1[s.n], 2), 'm/s'
              '\n    x2=', round(s.x2[s.n], 2), 'm | v2 =', round(s.v2[s.n], 2), 'm/s'
              '\n    Distance climbed:', round(climbDistance / 1000, 2), 'km',
              '\n    time taken to climb:', round(climbTime, 2), 's',
              '\n    average climb rate:', round(avgCR, 2), 'm/s')
        print('\n----------')


def Cruise(s, v1=30, distance=800, initialHeight='auto', fullOutput=True):
    """
    Sets the aircraft on a level flight path at a given height and runs
    for a given distance.
    """
    global completedDistance, initialx1, settledV1, settledx1, settledTime

    if distance / v1 < 15:
        msg = 'The distance passed to the Cruise is too short. The flight ' \
            + 'would roughly take (' + str(round(distance, 2)) + ' m)/(' \
            + str(round(v1, 2)) + ' m/s) = ' + str(round(distance / v1, 2)) \
            + ' s, but at least 15 seconds are needed for the aicraft '\
            + 'to settle into the cruise.'
        raise ValueError(msg)

    # save initial displacement value for later
    initialx1 = s.x1[s.n]
    initialTime = s.time

    if initialHeight == 'auto':
        initialHeight = s.x2[s.n]

    # Needed thrust and angle of attack needed to fly at given velocity
    T, alpha = LevelFlight(s, initialHeight, v1)

    # convert to int
    T = T[0]
    alpha = alpha[0]

    # pitch and alpha are the same during cruise
    pitch = alpha

    if fullOutput:
        print('\n----------')
        print('-- CRUISE --')
        print('Starting cruise at t=', s.time,
              '\n    x1=', round(s.x1[s.n], 2), 'm | v1 =', round(s.v1[s.n], 2), 'm/s',
              '\n    x2=', round(s.x2[s.n], 2), 'm | v2 =', round(s.v2[s.n], 2), 'm/s',
              '\n    desired distance =', round(distance / 1000, 2), ' km | desired v =',
              round(v1, 2), 'm/s',
              '\n    Thrust setting =', round(T, 2), 'N | pitch setting =',
              round(pitch, 2), 'degrees')
        print('\n')


    ########## START ##########
    # set inputs using sigmoids
    turnDuration = 10 * abs(AC['maxTurnRate'] / (pitch - s.pitch[s.n]))
    
    s.TShape = 'sigmoid';             s.pitchShape = 'sigmoid'
    s.TFloor = s.thrust[s.n];         s.pitchFloor = s.pitch[s.n]
    s.TCeiling = T;                   s.pitchCeiling = pitch
    s.TStart = 0;                     s.pitchStart = 0
    s.TEnd = 2;                       s.pitchEnd = turnDuration

    # run the simulation until it settles
    s.runtime = 15
    s.steps = 5
    s.RunInterval()
    
    ########## END ##########
    
    # record some values, now that the cruise is settled
    settledV1 = s.v1[s.n]
    settledTime = s.time
    settledx1 = s.x1[s.n]
    
    ########## START ##########
    
    h = s.x2[s.n]

    # readjust control inputs
    T, alpha = LevelFlight(s, initialHeight, v1)

    s.TShape = 'sigmoid';             s.pitchShape = 'sigmoid'
    s.TFloor = s.thrust[s.n];         s.pitchFloor = s.pitch[s.n]
    s.TCeiling = T;                   s.pitchCeiling = pitch
    s.TStart = 0;                     s.pitchStart = 0
    s.TEnd = 3;                       s.pitchEnd = 3
    
    completedDistance = s.x1[s.n] - initialx1
    s.runtime = (distance - completedDistance) / s.v1[s.n]
    s.steps = 10
    s.RunInterval()
    
    ########## END ##########
    
    ########## START ##########
    # distance already completed for the cruise
    completedDistance = s.x1[s.n] - initialx1
    while distance - completedDistance > 5 * s.v1[s.n]:
        # run until the end
        s.runtime = 5
        # print('Distance missing: ', distance - completedDistance, 'm',
        #      '\nvelocity: ', s.v1[s.n], 'm/s \n runtime =', s.runtime)
        s.steps = 5
        s.RunInterval()
        
        completedDistance = s.x1[s.n] - initialx1
        
    ########## END ##########
    
    # values for checking and printing
    cruiseDistance = s.x1[s.n] - initialx1
    cruiseTime = s.time - initialTime
    avgV1 = cruiseDistance / cruiseTime
    heightError = s.x2[s.n] - initialHeight
    
    # print warning if the operator drifted off the correct height (within a
    # margin of 10%)
    if abs((heightError) / initialHeight) > 0.1:
        print('\nWARNING: THE AIRCRAFT DRIFTED OFF THE CORRECT HEIGHT WHILE CRUISING.',
              '\nTHE CORRECT HEIGHT IS ', initialHeight, 'm BUT THE CRUISE WAS',
              'ENDED AT', s.x2[s.n], 'm (v1=', round(v1, 3),
              'm/s, Distance=', round(distance / 1000, 3), 'km)\n')
    
    # print warning if the operator did not cruise for the desired distance (within a
    # margin of 10%)
    if abs((cruiseDistance - distance) / distance) > 0.15:
        print('\nWARNING: THE CRUISE DID NOT COVER THE CORRECT DISTANCE.',
              '\nTHE DESIRED DISTANCE IS ', distance, 'm BUT THE ACHIEVED',
              'DISTANCE IS', cruiseDistance, 'm (Alt=', round(s.x2[s.n] / 1000, 3),
              'km, v1=', round(v1, 3), 'm/s)\n')
        
    # print warning if the operator did not achieve the correct velocity
    # (within a margin of 10%)
    if abs((s.v1[s.n] - v1) / v1) > 0.1:
        print('\nWARNING: THE CRUISE DID NOT ACHIEVE THE CORRECT VELOCITY.',
              'THE DESIRED VELOCITY IS ', v1, 'm BUT THE ACHIEVED',
              'VELOCITY IS', avgV1, 'm/s (Alt=', round(s.x2[s.n] / 1000, 3),
              'km,  Distance=', round(distance / 1000, 3), 'km)\n')
    
    
    if fullOutput:
        print('Finished cruise at t=', s.time,
              '\n    x1=', round(s.x1[s.n], 2), 'm | v1 =', round(s.v1[s.n], 2), 'm/s'
              '\n    x2=', round(s.x2[s.n], 2), 'm | v2 =', round(s.v2[s.n], 2), 'm/s'
              '\n    Distance cruised:', round(cruiseDistance / 1000, 2), 'km',
              '\n    time taken to cruise:', round(cruiseTime, 2), 's',
              '\n    average velocity:', round(avgV1, 2), 'm/s')
        print('\n----------')


def Glide(s, desiredHeight, fullOutput=True):
    """ makes the aircraft glide at the minimum glide angle until
    the desired height is reached"""
    
    # save values for later
    initialx1 = s.x1[s.n]
    initialHeight = s.x2[s.n]
    initialTime = s.time
    
    # minimum glide angle occurs at highest L/D
    alpha, LDRatio = ClCd(h=s.x2[s.n])
    
    gamma = math.atan2(1, LDRatio)
    
    pitch = alpha + gamma
    
    if fullOutput:
        print('\n----------')
        print('-- GLIDE --')
        print('Starting glide at t=', s.time,
              '\n    x1=', round(s.x1[s.n], 2), 'm | v1 =', round(s.v1[s.n], 2), 'm/s',
              '\n    x2=', round(s.x2[s.n], 2), 'm | v2 =', round(s.v2[s.n], 2), 'm/s',
              '\n    Glide angle =', round(gamma, 2), 'degrees | pitch for glide =',
              round(pitch, 2), 'degrees',
              '\n    thrust:', round(s.thrust[s.n], 2), 'N')
        print('\n')
    
    
    ##########
    turnDuration = 5 * abs(AC['maxTurnRate'] / (pitch - s.pitch[s.n]))
    s.TShape = 'sigmoid';             s.pitchShape = 'sigmoid'
    s.TFloor = s.thrust[s.n];         s.pitchFloor = s.pitch[s.n]
    s.TCeiling = 0;                   s.pitchCeiling = pitch
    s.TStart = 0;                     s.pitchStart = 0
    s.TEnd = 3;                       s.pitchEnd = turnDuration
    
    # let it settle
    s.runtime = 30
    s.steps = 5
    s.RunInterval()
    
    ##########
    
    ##########
    # find remaining flight time to reach the desired height, because
    # descent rate changes with height
    t = abs((s.x2[s.n] - desiredHeight) / s.v2[s.n])
    # fly 70 % of that time
    s.runtime = t * 0.7
    s.steps = 100
    s.RunInterval()
    ##########
    
    ##########
    # find remaining flight time to reach the desired height
    t = abs((s.x2[s.n] - desiredHeight) / s.v2[s.n])
    # fly to desired height
    s.runtime = t
    s.steps = 100
    s.RunInterval()
    ##########
    
    
    # values for checking and printing
    glideDistance = s.x1[s.n] - initialx1
    glideHeight = s.x2[s.n] - initialHeight
    glideTime = s.time - initialTime
    avgDR = glideHeight / glideTime
    avgv1 = glideDistance / glideTime
    
    # print warning if the operator did not descend thtough the desired altitude
    # (within a margin of 10%)
    if 1 - abs(glideHeight / (initialHeight - desiredHeight)) > 0.1:
        print('\nWARNING: THE GLIDE DID NOT GO TO THE THE CORRECT ALTITUDE.',
              '\nTHE DESIRED ALTITUDE IS ', desiredHeight, 'm BUT THE ACHIEVED \
              ALTITUDE IS', s.x2[s.n], 'm \n')
    
    if fullOutput:
        print('Finished glide at t=', s.time,
              '\n    x1=', round(s.x1[s.n], 2), 'm | v1 =', round(s.v1[s.n], 2), 'm/s'
              '\n    x2=', round(s.x2[s.n], 2), 'm | v2 =', round(s.v2[s.n], 2), 'm/s'
              '\n    Distance glided:', round(glideDistance / 1000, 2), 'km',
              '\n    time taken to glide:', round(glideTime, 2), 's',
              '\n    average horizontal velocity:', round(avgv1, 2), 'm/s',
              '\n    average descent rate:', round(avgDR, 2), 'm/s',
              '\n    thrust:', round(s.thrust[s.n], 2), 'N')
        print('\n----------')
        
        

def Drop(s, initialHeight):
    s.y0 = [0, initialHeight, 0, 0]
    s.x2[s.n] = initialHeight
    s.TShape = 'const'
    s.TCeiling = 0
    
    s.runtime = 2
    s.steps = 2
    s.RunInterval()

    while s.x2[s.n] > 0:
        s.RunInterval()


def Hover(s, initialHeight=100):
    s.y0 = [0, initialHeight, 0, 0]
    s.pitchShape = 'const'
    s.pitchCeiling = math.pi / 2
    s.TShape = 'const'
    s.TCeiling = s.totalMass * AC['g']
    
    s.runtime = 10
    s.steps = 10
    s.RunInterval()
