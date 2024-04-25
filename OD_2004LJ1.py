# SSP 2021 OD Code
# 07/24/2021
# Alex Puga

# This program takes a file containing the position of any terrestrial satellite, e.g. an asteroid,
# at three different times, and determines an orbital trajectory for that body

from math import sqrt, sin, cos, radians, asin, acos, pi
import numpy as np
from numpy.lib import imag
from astropy.io import fits
import matplotlib.pyplot as plt

def findQuadrant(sine, cosine):
    if cosine > 0 and sine > 0: #1
        return asin(sine)

    if cosine < 0 and sine > 0: #2
        return acos(cosine)

    if cosine < 0 and sine < 0: #3
        return pi - asin(sine)

    if cosine > 0 and sine < 0: #4
        return 2*pi + asin(sine)

def prelimOD(r, rdot, t, to):
    # variables and constants
    r = np.array(r)
    rdot = np.array(rdot)

    mu = (0.0172020989484)**2
    tau = sqrt(mu) * t

    x, y, z = r
    xdot, ydot, zdot = rdot

    # determine the h vector
    h = np.cross(r, rdot)
    hx, hy, hz = h
    hMag = sqrt(hx**2 + hy**2 + hz**2)

    # determine a
    rMag = sqrt(x**2 + y**2 + z**2)
    a = ((2 / rMag) - np.dot(rdot, rdot))**-1

    # determine e
    e = (1 - (np.linalg.norm(np.cross(r, rdot))**2 / a))**(1/2)
    
    # determine I
    rcrossrdot = np.cross(r, rdot)
    rcrossrdotx, rcrossrdoty, rcrossrdotz = rcrossrdot
    I = acos(rcrossrdotz / np.linalg.norm(np.cross(r, rdot)))

    # determine sigma
    omegaSIN = hx / (hMag * sin(I))
    omegaCOS = -hy / (hMag * sin(I))

    omega = findQuadrant(omegaSIN, omegaCOS)

    # determine w
    fwSIN = z / (rMag * sin(I))
    fwCOS = 1/cos(omega) * ((x / rMag) + (cos(I) * fwSIN * sin(omega)))
    fw = findQuadrant(fwSIN, fwCOS)

    fCOS = (rMag / e) * ((a * (1 - e**2) / rMag) - 1)
    fSIN = (np.dot(r, rdot) / (e * rMag)) * sqrt(a * (1 - e**2))
    f = findQuadrant(fSIN, fCOS)
    w = fw - f

    # determine m M at time of observation
    E = acos((x/a) + e)
    E2 = acos((1 / e) * (1 - (rMag / a)))

    if f > pi:
        E2 = -1 * E2

    M2 = E2 - (e * sin(E2))
    n = sqrt(mu / a**3)

    Mo = M2 + (n * (to - t))

    I = I * 180/pi
    omega = omega * 180/pi
    w = w * 180/pi % 360
    Mo = Mo * 180/pi % 360
    
    return a, e, I, w, omega, Mo

def SEL(taus, Sun2, rhohat2, D0, D21, D22, D23):
    # initiate necessary return variables and constants
    mu = 1
    r2 = 0
    roots = []
    rhos = []

    # set vector arrays as numpy arrays
    Sun2 = np.array(Sun2)
    rhohat2 = np.array(rhohat2)

    # read in arrays and make individual assignments to access components of vectors
    tau1, tau3, tau = taus
    Sunx, Suny, Sunz = Sun2
    rhohatx, rhohat2y, rhohat2z = rhohat2

    # begin with a the truncated f and g series, and
    # algebraically calculate A1, B1, A3, and B3
    A1 = tau3 / tau
    B1 = (A1 * (tau**2 - tau3**2)) / 6

    A3 = -tau1 / tau
    B3 = (A3 / 6 )* (tau**2 - tau1**2)

    # using those in the scalar equation for rho2 from eq(101),
    # get an equation which relates rho2 and r2
    A = -((A1 * D21) - D22 + (A3 * D23))/ D0
    B = -((B1 * D21) + (B3 * D23)) / D0

    ### rho2 = A + ((mu * B) / (r2 **3))

    # use eq(2) to derive an equation to solve for r2
    # take the dot product of eq(2) with itself:
    E = -2 * np.dot(rhohat2, Sun2)
    F = Sunx**2 + Suny**2 + Sunz**2

    ### r2 = math.sqrt((rho2 ** 2) + (E * rho2) + F)

    # using eq(128) to substitute for rho2, we get
    # an 8th order polynomial in r2
    a = -1 * (A**2 + (A * E) + F)
    b = -1 * mu * ((2 * A * B) + (B * E))
    c = -1 * mu**2 * B**2

    # scalar equation of Lagrange
    # r2**8 + (a * r2**6) + (b * r2**3) + c = 0

    # solve for all roots
    allRoots = np.polynomial.polynomial.polyroots([c, 0, 0, b, 0, 0, a, 0, 1])

    # iterate to get only positive real roots
    for i in range(0, len(allRoots)):
        if allRoots[i].real > 0 and allRoots[i].imag ==0:
            roots.append(allRoots[i].real)

    # corresponding range value rho2 for each possible r2
    for i in range(0, len(roots)):
        rho = A + ((mu * B)/(roots[i]**3))
        rhos.append(rho)

    return roots, rhos

def fg(tau, r2, r2dot, flag):
    # assign individual components
    r2 = np.array(r2)
    r2dot = np.array(r2dot)
    r2x, r2y, r2z = r2
    r2dotx, r2doty, r2dotz = r2dot

    # constants
    mu = 1

    # set gausssian time interval
    Ti = tau

    # begin by expanding the position vector ri in a taylor series
    # about the central value r2
    r2Mag = sqrt(r2x**2 + r2y**2 + r2z**2)

    u = mu / r2Mag**3
    z = np.dot(r2, r2dot) / r2Mag**2
    q = (np.dot(r2dot, r2dot) / r2Mag**2) - u

    r2dot2 = (-mu * r2) / (r2Mag **3)
    r2dot3 = (-mu / r2Mag **5) * (((r2 **2) * r2dot) - (3 * np.dot(r2, r2dot) * r2))
    r2dot4 = (((3 * u * q) - (15 * u * z**2) + u**2) * r2) + (6 * u * z * r2dot)

    # f and g series (third order)
    f3 = 1 - ((u * Ti**2) / 2) + ((u * z * Ti**3) / 2)
    g3 = Ti - ((u * Ti**3) / 6)
    
    # f and g series (fourth order)
    f4 = 1 - ((u * Ti**2) / 2) + ((u * z * Ti**3) / 2) + ((((3 * u * q) - (15 * u * z**2) + u**2) * Ti**4) / 24)
    g4 = Ti - ((u * Ti**3) / 6) + ((u * z * Ti**4) / 4)

    # use the values from eq.(102) and group terms by r2 and r2dot
    # r = (f * r2) + (g * r2dot)

    # find deltaE and other variables
    a = ((2 / r2Mag) - (np.dot(r2dot, r2dot) / mu))**-1
    n = sqrt(mu / a**3)

    # iterate to determine deltaE
    e = sqrt(1 - (np.linalg.norm(np.cross(r2, r2dot))**2) / (mu * a))
    const = (0.85 * e) - (np.dot(r2, r2dot) / (n * a**2))
    rdotprod = np.dot(r2, r2dot) / (n * a**2)
    sign = (rdotprod  * cos(n * Ti - rdotprod)) + ((1 - (r2Mag / a)) * sin(n * Ti - rdotprod))

    if e < 0.1:
        Xo = n * Ti
    else:
        if sign > 0:
            Xo = n * Ti + const
        else:
            Xo = n * Ti - const 
    
    x = Xo
    fx = x - ((1 - (r2Mag/a)) * sin(x)) + rdotprod * (1 - cos(x)) - (n * Ti)
    xprime = 1 - ((1 - r2Mag / a) * cos(x)) + (np.dot(r2, r2dot) / (n * a**2)) * sin(x)
    x2 = x - fx/xprime

    while True:
        x = x2
        fx = x - ((1 - (r2Mag/a)) * sin(x)) + rdotprod * (1 - cos(x)) - (n * Ti)
        xprime = 1 - ((1 - r2Mag / a) * cos(x)) + (np.dot(r2, r2dot) / (n * a**2)) * sin(x)

        x2 = x - fx/xprime

        if (abs(x - x2) < 1e-12):
            break

    # f and g functions
    deltaE = x2
    fi = 1 - ((a / r2Mag) * (1 - cos(deltaE)))
    gi = Ti + ((1 / n) * (sin(deltaE) - deltaE))

    if flag == 1:
        return f3, g3

    if flag == 2:
        return f4, g4

    if flag == 3:
        return fi, gi

def MoG(fileOpen):
    # constants
    # Gaussian gravitational constant
    k = 0.0172020989484

    # speed of light in au/(mean solar)day
    cAU = 173.144643267

    # Earth's obliquity
    eps = radians(23.4374)
    
    # mu in Gaussian
    mu = 0.0172020989484 ** 2

    file = open(fileOpen)
    data = []

    for line in file.readlines():
        # split each line at white spaces
        # will return a list with the four input values
        row = line.split()

        # first two elements correspond to the x and y pixel values
        t = float(row[0])

        # last two elements correspond to the ra and dec
        Ra = row[1]
        Dec = row[2]
        sunx = row[3]
        suny = row[4]
        sunz = row[5]

        # split row at colons for converting from hms to degrees
        ra = Ra.split(':')
        h = float(ra[0])
        mm = float(ra[1])
        ss = float(ra[2])

        dec = np.array(Dec.split(':')).astype(float)
        dec = np.copysign(dec, dec[0])
        h2 = float(dec[0])
        mm2 = float(dec[1])
        ss2 = float(dec[2])

        # calculate RA and DEC
        Ra = radians(float((h * 15) + (mm * 0.25) + (ss * (0.25/60))))
        Dec = radians(float(h2 + mm2/60 + ss2/3600))

        # append data to array
        data.append([t, Ra, Dec, sunx, suny, sunz])
    
    data1 = data[0]
    data2 = data[1]
    data3 = data[2]

    # individual vector components
    t1, ra1, dec1, sunx1, suny1, sunz1 = data1
    sun1 = np.array([sunx1, suny1, sunz1])
    sun1 = sun1.astype(float)

    t2, ra2, dec2, sunx2, suny2, sunz2 = data2
    sun2 = np.array([sunx2, suny2, sunz2])
    sun2 = sun2.astype(float)

    t3, ra3, dec3, sunx3, suny3, sunz3 = data3
    sun3 = np.array([sunx3, suny3, sunz3])
    sun3 = sun3.astype(float)

    # determine the Gaussian time intervals
    tau3 = k * (t3 - t2)
    tau1 = k * (t1 - t2)
    tau = tau3 - tau1
    taus = [tau1, tau3, tau]

    # calculate rhohat unit vectors
    rho1hat = np.array([cos(ra1)*cos(dec1), sin(ra1)*cos(dec1), sin(dec1)])
    rho1hat1, rho1hat2, rho1hat3 = rho1hat

    rho2hat = np.array([cos(ra2)*cos(dec2), sin(ra2)*cos(dec2), sin(dec2)])
    rho2hat1, rho2hat2, rho2hat3 = rho2hat

    rho3hat = np.array([cos(ra3)*cos(dec3), sin(ra3)*cos(dec3), sin(dec3)])
    rho3hat1, rho3hat2, rho3hat3 = rho3hat

    # scalar equations of range
    Do = np.dot(rho1hat, np.cross(rho2hat, rho3hat))
    D11 = np.dot(np.cross(sun1, rho2hat), rho3hat)
    D12 = np.dot(np.cross(sun2, rho2hat), rho3hat)
    D13 = np.dot(np.cross(sun3, rho2hat), rho3hat)

    D21 = np.dot(np.cross(rho1hat, sun1), rho3hat)
    D22 = np.dot(np.cross(rho1hat, sun2), rho3hat)
    D23 = np.dot(np.cross(rho1hat, sun3), rho3hat)

    D31 = np.dot(rho1hat, np.cross(rho2hat, sun1))
    D32 = np.dot(rho1hat, np.cross(rho2hat, sun2))
    D33 = np.dot(rho1hat, np.cross(rho2hat, sun3))

    r2Guess, rhos = SEL(taus, sun2, rho2hat, Do, D21, D22, D23)

    print(r2Guess)
    print(rhos)

    #user-prompted to select root to use
    r2Guess = input("Choose an initial value for r2:")
    r2Guess = float(r2Guess)

    # determine initial truncated values for f1, f3, g1, g3
    mu1 = 1

    f1 = 1 - mu1 * tau1**2 / (2 * r2Guess**3)
    g1 = tau1 - mu1 * tau1**3 / (6 * r2Guess**3)
    f3 = 1 - mu1 * tau3**2 / (2 * r2Guess**3)
    g3 = tau3 - mu1*tau3**3 / (6 * r2Guess**3)

    # calculate c1 and c3
    c1 = g3 / ((f1 * g3) - (g1 * f3))
    c3 = -g1 / ((f1 * g3) - (g1 * f3))
    c2 = -1

    # d's 
    d1 = -f3 / ((f1 * g3) - (f3 * g1))
    d3 = f1 / ((f1 * g3) - (f3 * g1))

    rho1 = (c1 * D11 + c2 * D12 + c3 * D13) / (c1 * Do)
    rho2 = (c1 * D21 + c2 * D22 + c3 * D23) / (c2 * Do)
    rho3 = (c1 * D31 + c2 * D32 + c3 * D33) / (c3 * Do)

    r1 = rho1 * np.array(rho1hat) - sun1
    r3 = rho3 * np.array(rho3hat) - sun3

    r2dot = d1 * r1 + d3  * r3
    r2 = rho2 * np.array(rho2hat) - sun2
    r2x, r2y, r2z = r2
    r2Mag = sqrt(r2x**2 + r2y**2 + r2z**2)

    u = 1 / (r2Mag**3)
    a = (2 / r2Mag - np.dot(r2dot, r2dot))**-1
    e = sqrt(1 - np.linalg.norm(np.cross(r2, r2dot))**2 / a)

    # correct for light travel time
    t1new = t1 - rho1/cAU
    t2new = t2 - rho2/cAU
    t3new = t3 - rho3/cAU

    tau3 = k * (t3new - t2new)
    tau1 = k * (t1new - t2new)
    tau = tau3 - tau1
    taus = [tau1, tau3, tau]
    
    count = 0

    while True:
        rho2First = rho2
        f1, g1 = fg(tau1, r2, r2dot, 3)
        f3, g3 = fg(tau3, r2, r2dot, 3)

        # calculate c1 and c3
        c1 = g3 / ((f1 * g3) - (g1 * f3))
        c3 = -g1 / ((f1 * g3) - (g1 * f3))
        c2 = -1

        d1 = -f3 / ((f1 * g3) - (f3 * g1))
        d3 = f1 / ((f1 * g3) - (f3 * g1))

        # calculate rhos
        rho1 = (c1 * D11 + c2 * D12 + c3 * D13) / (c1 * Do)
        rho2 = (c1 * D21 + c2 * D22 + c3 * D23) / (c2 * Do)
        rho3 = (c1 * D31 + c2 * D32 + c3 * D33) / (c3 * Do)

        r1 = rho1 * np.array(rho1hat) - sun1
        r3 = rho3 * np.array(rho3hat) - sun3

        r2dot = d1 * r1 + d3 * r3
        r2 = rho2 * np.array(rho2hat) - sun2
        r2x, r2y, r2z = r2
        r2Mag = sqrt(r2x**2 + r2y**2 + r2z**2)

        u = 1 / (r2Mag**3)
        a = (2 / r2Mag - np.dot(r2dot, r2dot))**-1
        e = sqrt(1 - np.linalg.norm(np.cross(r2, r2dot))**2 / a)
        
        if abs(rho2First - rho2) < 1E-12:
            break
        count = count + 1

        # correct for light travel time
        t1New = t1 - rho1/cAU
        t2New = t2 - rho2/cAU
        t3New = t3 - rho3/cAU

        tau3 = k * (t3New - t2New)
        tau1 = k * (t1New - t2New)
        tau = tau3 - tau1
        taus = [tau1, tau3, tau]

    # convert from equatorial to ecliptic coordinates
    # by rotating by obliquity
    R2X = r2[0]
    R2Y = (cos(eps) * r2[1]) + (sin(eps) * r2[2])
    R2Z = -(sin(eps)) * r2[1] + (cos(eps) * r2[2])
    
    r2New = np.array([R2X, R2Y, R2Z])

    R2DOTX = r2dot[0]
    R2DOTY = (cos(eps) * r2dot[1]) + (sin(eps) * r2dot[2])
    R2DOTZ = -(sin(eps)) * r2dot[1] + (cos(eps) * r2dot[2])

    r2dotNew = np.array([R2DOTX, R2DOTY, R2DOTZ])

    # epoch time
    to = 2459419.79167
    a, e, I, w, omega, Mo = prelimOD(r2New, r2dotNew, t2New, to)
        
    print("a: ", a)
    print("e: ", e)
    print("i: ", I)
    print("omega: ", w)
    print("Omega: ", omega)
    print("M: ", Mo)

print(MoG('input.txt'))
