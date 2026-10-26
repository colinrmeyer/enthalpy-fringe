from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve
import time

class meshinformation():
    def __init__(self):
        # # mesh information # #
        # domain from from z=l to z=r
        self.l = 0 # nondimensional bottom
        self.r = 12 # nondimensional top
        self.N = 256 # Number of grid cells
        self.dy = (self.r-self.l)/self.N # grid cell width
        self.yedges = np.linspace(self.l,self.r,self.N+1) # cell edges
        self.y = (self.yedges[:self.N]+self.yedges[-self.N:])/2 # cell centers
        # # time information # #
        self.TotalTime = 3 # total nondimensional time

class parameters():
    def __init__(self,effpval):
        # # parameters # #
        self.rhoi = 917; # ice density, kg/m^3
        self.rhow = 1000; # water density, kg/m^3
        self.rhos = 2500; # sediment density, kg/m^3
        self.cpi = 2050; # heat capacity of ice, J/(kg*degC)
        self.cpw = 4186; # heat capacity of water, J/(kg*degC)
        self.Ki = 2.1; # ice thermal conductivity, W/(m*degC)
        self.Kw = 0.56; # water thermal conductivity, W/(m*degC)
        self.Ks = 1.4; # sediment thermal conductivity, W/(m*degC)
        self.Latent = 3.34e5; # latent heat of fusion, J/kg
        self.g = 9.806; # acceleration due to gravity, m/s^2
        self.qin = 0.717704528982468; # geothermal heat flux, W/m^2
        self.Tm = 273.15; # melting temperature, Kelvin
        self.k0 = 9.48e-13; # permeability, m^2
        self.gamma = 0.034; # interfacial energy, J/m^2
        self.mu = 1.8e-3; # viscosity of water, Pa*s
        self.rp = 78.2630e-6; # grain size, m
        self.alpha = 7.36; # exponent in permeability as a function of temperature
        self.beta = 1.6; # exponent in saturation as a function of temperature
        self.EffP = effpval; # effective pressure, Pa
        self.sigman = 1e6; # overburden pressure, Pa
        self.phi = 0.403; # porosity
        self.y2s = 3600*365*24; # 1 year in seconds, s
        self.d= 10e-9 # film thickness, m
        self.R = 100e-6 # grain radius, m
        self.f = 1 # film shape factor
        # # derived parameters # #
        self.EffPc = self.gamma*(2/self.rp) # fringe entrance pressure, Pa
        self.Tf = self.Tm-((self.EffPc*self.Tm)/(self.rhoi*self.Latent)) # fringe entrance temperature, K 

class scales():
    def __init__(self,effpval):
        prms = parameters(effpval)
        self.K = prms.Ki
        self.k = prms.k0; # permeability scale, m^2
        self.EffP = prms.EffPc; # effective pressure scale, Pa
        self.q = prms.qin; # heat flux scale, K/m
        self.T = prms.Tm*self.EffP/(prms.rhoi*prms.Latent); # temperature difference scale, degC
        self.z = self.K*self.T/self.q; # distance scale, m
        self.V = self.k*self.EffP/(prms.mu*self.z); # heave rate scale, m/s
        self.t = prms.rhoi*prms.Latent*(self.z**2)/(self.K*self.T); # timescale, s
        
class nondimensional():
    def __init__(self,effpval,heaverate):
        prms = parameters(effpval)
        bracket = scales(effpval)
        self.delta = 1-(prms.rhoi/prms.rhow); # scaled ice/water density difference
        self.nu = (prms.rhos/prms.rhow); # sediment to water density ratio
        self.Pe = bracket.V*bracket.t/bracket.z; # Peclet number
        self.G = prms.rhow*prms.g*bracket.z/bracket.EffP; # fringe hydrostatic pressure
        self.Stefan = prms.Latent/(prms.cpi*bracket.T); # Stefan number
        self.EffP = prms.EffP/bracket.EffP; # nd eff p
        self.V = heaverate/bracket.V; # nd heave rate (set as a parameter in m/s)
        self.Pi0 = (prms.k0*(prms.R**2)*prms.f)/(bracket.z*(prms.d**3)) # film pressure

@jit(nopython=True)
def enthalpy2temperature(H,N,phi,Stefan,beta):
    T = np.zeros(N)
    for i in range(N):
        if H[i]>0:
            T[i] = -Stefan*(H[i]/phi)
        elif H[i]<=0 and H[i]>-phi:
            T[i]=((1+(H[i]/phi))**(-1/beta))-1
        else:
            T[i]=-Stefan*(H[i]+1)
    return T

@jit(nopython=True)
def temperature2enthalpy(T,Tl,N,phi,Stefan,beta):
    H = np.zeros(N)
    for i in range(N):
        if T[i]<0:
            H[i] = -(phi*T[i])/Stefan
        elif T[i]>=0 and T[i]<Tl:
            H[i] = -phi*Sfun(T[i],beta)
        else:
            H[i] = -(T[i]/Stefan)-1
    return H

@jit(nopython=True)
def Sfun(T,beta):
    return 1-((1+(T>0)*T)**(-beta))

@jit(nopython=True)
def kfun(T,alpha):
    return (1+(T>0)*T)**(-alpha)

@jit(nopython=True)
def Kfun(T,Ks,Ki,Kw,K,phi,beta,alpha):
    return (Ks**(1-phi))*(Ki**(phi*Sfun(T,beta)))*(Kw**((1-Sfun(T,beta))*phi))/K

@jit(nopython=True)
def Vintegrals(theta,heave,alpha,beta,G,nu,phi,y,dy,EffP):
    It1 = G*(nu-1)*(1-phi)*(y[heave][-1]-y[~heave][-1])
    It2 = (1-phi)*theta[-1] + (phi/(1-beta))*(((1+theta[-1])**(1-beta))-1)
    Ib = np.sum((((1-phi*Sfun(theta,beta))**2)/kfun(theta,alpha))*dy*heave)
    return (1-EffP+It1+It2)/Ib

@jit(nopython=True)
def VintegralsFilm(theta,heave,alpha,beta,G,nu,phi,y,dy,EffP,Pi):
    h   = np.sum(heave)*dy # fringe weight
    It1 = G*(nu-1)*(1-phi)*h
    thl = theta[-1]; tp = max(thl,0.0) # pore ice only if theta_l > 0
    It2 = thl - phi*tp + (phi/(1-beta))*(((1+tp)**(1-beta))-1)
    Ib = np.sum((((1-phi*Sfun(theta,beta))**2)/kfun(theta,alpha))*dy*heave)
    return (1-EffP+It1+It2)/(Ib+Pi)

@jit(nopython=True)
def EffPloc(theta,heave,alpha,beta,G,nu,phi,y,dy,EffP,velocity):
    IG = G*np.cumsum((nu*(1-phi)+phi*(1-Sfun(theta,beta)))*dy*heave)
    Ipsdt = phi*theta*heave + (phi/(1-beta))*(((1+theta*heave)**(1-beta))-1) + phi*Sfun(theta*heave,beta)*(1+theta*heave)
    Iv = velocity*np.cumsum((((1-phi*Sfun(theta,beta))**2)/kfun(theta,alpha))*dy*heave)
    return EffP*heave-IG-Ipsdt+Iv

def fringe_thickness(H,mesh,prms,nd):
    # h = zl - zf, 
    # zf from linear interpolation of theta = 0; zero if no fringe
    if H[-1] > 0:
        return 0.0
    th = enthalpy2temperature(H,mesh.N,prms.phi,nd.Stefan,prms.beta)
    unfrozen = np.nonzero(H > 0)[0]
    if unfrozen.size == 0:
        return mesh.r - mesh.l
    i = unfrozen[-1] # top unfrozen cell; i+1 is lowest fringe cell
    zf = mesh.y[i] + (0 - th[i])/(th[i+1] - th[i])*mesh.dy
    return mesh.r - zf

@jit(nopython=True)
def enthalpyode(time,ye,N,y,dy,phi,Stefan,alpha,beta,G,nu,EffP,Pe,V,Ks,Ki,Kw,K,Pi0):
        theta = enthalpy2temperature(ye,N,phi,Stefan,beta)
        # impose force balance velocity
        heaving = ye<=0
        Pi = Pi0 # thin film correction
        velocity_fb = Pe*VintegralsFilm(theta,heaving,alpha,beta,G,nu,phi,y,dy,EffP,Pi)
        # conservative upwind advection of the ice (latent) enthalpy, one flux per face:
        #   F_{i+1/2} = Vhat * Hice_upwind,  Hice = min(H,0) = -phi*S in fringe, 0 in unfrozen till
        # afm[i+1] = afp[i] so interior fluxes cancel exactly; flux -> 0 continuously at the fringe base
        Hice = np.minimum(ye,0.0)
        if velocity_fb > 0:
            afp = velocity_fb*Hice                                # face i+1/2, upwind cell i
        else:
            afp = velocity_fb*np.append(Hice[1:],Hice[-1])        # face i+1/2, upwind cell i+1
        afp[-1] = velocity_fb*Hice[-1]                            # lens face; total top flux set by dfp[-1]
        afm = np.append(velocity_fb*Hice[0], afp[:-1])            # bottom face, then shared faces
        
        Kc = Kfun(theta,Ks,Ki,Kw,K,phi,beta,alpha) # center
        Kp = np.append(Kc[1:],Kc[-1]) # positive
        Km = np.append(Kc[0],Kc[:-1]) # minus
        thetap = np.append(theta[1:],theta[-1])
        thetam = np.append(theta[0],theta[:-1])
        dfp = (2/dy)*((Kp*Kc)/(Kp+Kc))*(thetap-theta)
        dfm = (2/dy)*((Kc*Km)/(Kc+Km))*(theta-thetam)
        dfm[0]= 1
        # total flux through lens face = q_ice - V (exact, with or without a fringe):
        # with a fringe afp[-1] = V*ye[-1] and this equals the old (1+PeV) - V(1+ye[-1]);
        # with no fringe the till below the lens is stationary, afp[-1] = 0, and q_f = q_ice - V
        dfp[-1] = (1 + Pe*V) - velocity_fb - afp[-1]
        fp = afp+dfp
        fm = afm+dfm
        return -((fp-fm)/dy)
        
if __name__ == "__main__":
    effectivepressure = 300*1000 # Pa 
    Vheaverate = -0.0262/(3600*365*24) # m/s
    
    mesh = meshinformation()
    prms = parameters(effectivepressure)
    bracket = scales(effectivepressure)
    nd = nondimensional(effectivepressure,Vheaverate)
    
    print(nd.Pi0)
    Pi_run = nd.Pi0
    # atol must be << phi/Stefan (~2e-6): unfrozen enthalpy H = -phi*theta/St is O(1e-6)
    tol = dict(rtol=1e-8, atol=1e-13)
    
    # # lowest ice lens height # #
    zl = mesh.r # top of domain
    
    # # initial temperature profile # #
    thetai = mesh.y-0.96*mesh.r
    thetalensi = 10
    enthalpyi = temperature2enthalpy(thetai,thetalensi,mesh.N,prms.phi,nd.Stefan,prms.beta)
    
    # # run the model for steady state initial condition # #
    EffP_initial = 17.8
    sol0 = solve_ivp(lambda t,x: enthalpyode(t,x,mesh.N,mesh.y,mesh.dy,prms.phi,nd.Stefan,prms.alpha,prms.beta,nd.G,nd.nu,EffP_initial,nd.Pe,nd.V,prms.Ks,prms.Ki,prms.Kw,bracket.K,Pi_run),[0,mesh.TotalTime],enthalpyi,method='BDF',**tol)
    # # new initial condition # #
    enthalpy0 = sol0.y[:,-1]
    
    # # spin up characteristics # #
    h0 = fringe_thickness(enthalpy0,mesh,prms,nd)
    th0 = enthalpy2temperature(enthalpy0,mesh.N,prms.phi,nd.Stefan,prms.beta)
    v0 = VintegralsFilm(th0,enthalpy0<=0,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,EffP_initial,Pi_run)
    print(h0*bracket.z*100)
    
    # # run the model for real # #
    sol = solve_ivp(lambda t,x: enthalpyode(t,x,mesh.N,mesh.y,mesh.dy,prms.phi,nd.Stefan,prms.alpha,prms.beta,nd.G,nd.nu,nd.EffP,nd.Pe,nd.V,prms.Ks,prms.Ki,prms.Kw,bracket.K,Pi_run),[0,mesh.TotalTime],enthalpy0,method='BDF',**tol) # Radau, LSODA
        
    f = plt.figure()
    plt.plot(sol.y[:,0],mesh.y,'.-',color='grey')
    plt.plot(sol.y[:,-1],mesh.y,'k.-')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylim(0,zl)
    plt.ylabel('height, $z$')
    plt.xlabel('enthalpy, $H$')
    
    theta = enthalpy2temperature(sol.y[:,-1],mesh.N,prms.phi,nd.Stefan,prms.beta)
    V = VintegralsFilm(theta,sol.y[:,-1]<=0,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP,Pi_run)
    print(V) # V value
    
    f2 = plt.figure()
    plt.plot(theta,mesh.y,'k.-')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylim(0,zl)
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.ylabel(r'height, $z$', fontsize=16)
    plt.xlabel(r'temperature, $\theta$', fontsize=16)
    
    nt = np.shape(sol.y)[1]
    h = np.zeros(nt)   # fringe thickness, interpolated to theta = 0 (0 if no fringe)
    Vt = np.zeros(nt)  # heave rate seen by the ODE
    thl = np.zeros(nt) # lens temperature
    for i in range(nt):
        H = sol.y[:,i]
        th = enthalpy2temperature(H,mesh.N,prms.phi,nd.Stefan,prms.beta)
        h[i] = fringe_thickness(H,mesh,prms,nd)
        Vt[i] = VintegralsFilm(th,H<=0,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP,Pi_run)
        thl[i] = th[-1]
    print('spinup: h = %.4f cm, V = %.4g m/yr, theta_l = %.3f' % (h0*bracket.z*100, v0*bracket.V*prms.y2s, th0[-1]))
    print('final: h = %.4f cm, V = %.4g m/yr, theta_l = %.3f' % (h[-1]*bracket.z*100, Vt[-1]*bracket.V*prms.y2s, thl[-1]))
        
    f = plt.figure()
    plt.plot(sol.t*bracket.t/(3600*24),h*bracket.z*100,'k.',markersize=0.1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylabel(r'fringe thickness (cm)')
    plt.xlabel(r'time (days)')
    f.savefig("transient_fringe.pdf")