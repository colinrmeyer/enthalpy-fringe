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
        self.r = 1 # nondimensional top
        self.N = (2**5)*(self.r-self.l) # Number of grid cells
        self.dy = (self.r-self.l)/self.N # grid cell width
        self.yedges = np.linspace(self.l,self.r,self.N+1) # cell edges
        self.y = (self.yedges[:self.N]+self.yedges[-self.N:])/2 # cell centers
        # # time information # #
        self.TotalTime = 3 # total nondimensional time

class parameters():
    def __init__(self):
        # # parameters # #
        self.rhoi = 917; # ice density, kg/m^3
        self.rhow = 1000; # water density, kg/m^3
        self.rhos = 2500; # sediment density, kg/m^3
        self.cpi = 2050; # heat capacity of ice, J/(kg*degC)
        self.cpw = 4186; # heat capacity of water, J/(kg*degC)
        self.Ki = 2.1; # ice thermal conductivity, W/(m*degC)
        self.Kw = 0.56; # water thermal conductivity, W/(m*degC)
        self.Ks = 4.0; # sediment thermal conductivity, W/(m*degC)
        self.Latent = 3.34e5; # latent heat of fusion, J/kg
        self.g = 9.806; # acceleration due to gravity, m/s^2
        self.qin = 70e-3; # geothermal heat flux, 60 mW/m^2
        self.Tm = 273.15; # melting temperature, Kelvin
        self.k0 = 1e-17; # permeability, m^2
        self.gamma = 0.034; # interfacial energy, J/m^2
        self.mu = 1e-3; # viscosity of water, Pa*s
        self.rp = 1e-6; # grain size, m
        self.alpha = 3.1; # exponent in permeability as a function of temperature
        self.beta = 0.53; # exponent in saturation as a function of temperature
        self.EffP = 200*1000; # effective pressure, Pa
        self.sigman = 1e6; # overburden pressure, Pa
        self.phi = 0.35; # porosity
        self.y2s = 3600*365*24; # 1 year in seconds, s
        # # derived parameters # #
        self.EffPc = self.gamma*(2/self.rp) # fringe entrance pressure, Pa
        self.Tf = self.Tm-((self.EffPc*self.Tm)/(self.rhoi*self.Latent)) # fringe entrance temperature, K 

class scales():
    def __init__(self):
        prms = parameters()
        self.K = prms.Ki
        self.T = prms.Tm*prms.EffP/(prms.rhoi*prms.Latent); # temperature difference scale, degC
        self.q = prms.qin; # heat flux scale, K/m
        self.z = self.K*self.T/self.q; # distance scale, m
        self.k = prms.k0; # permeability scale, m^2
        self.EffP = prms.EffPc; # effective pressure scale, Pa
        self.V = self.k*self.EffP/(prms.mu*self.z); # heave rate scale, m/s
        self.t = prms.rhoi*prms.Latent*(self.z**2)/(self.K*self.T); # timescale, s
        
class nondimensional():
    def __init__(self):
        prms = parameters()
        bracket = scales()
        self.delta = 1-(prms.rhoi/prms.rhow); # scaled ice/water density difference
        self.nu = (prms.rhos/prms.rhow); # sediment to water density ratio
        self.Pe = bracket.V*bracket.t/bracket.z; # Peclet number
        self.G = prms.rhow*prms.g*bracket.z/bracket.EffP; # fringe hydrostatic pressure
        self.S = prms.Latent/(prms.cpi*bracket.T); # Stefan number
        self.Qout = 0.2; # qout/qin, nd flux through top
        self.EffP = prms.EffP/bracket.EffP; # nd eff p
        self.V = (self.Qout-1)/(prms.phi*self.Pe); # nd heave rate 

@jit(nopython=True)
def enthalpy2temperature(H,N,phi,S,beta):
    T = np.zeros(N)
    for i in range(N):
        if H[i]>phi:
            T[i] = S*((phi-H[i])/phi)
        elif H[i]<phi and H[i]>0:
            T[i]=((phi/H[i])**(1/beta))-1
        else:
            T[i]=-S*H[i]
    return T

@jit(nopython=True)
def Sfun(T,beta):
    return 1-((1+(T>0)*T)**(-beta))

@jit(nopython=True)
def kfun(T,alpha):
    return (1+(T>0)*T)**alpha

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
def EffPloc(theta,heave,alpha,beta,G,nu,phi,y,dy,EffP,velocity):
    IG = G*np.cumsum((nu*(1-phi)+phi*(1-Sfun(theta,beta)))*dy*heave)
    Ipsdt = phi*theta*heave + (phi/(1-beta))*(((1+theta*heave)**(1-beta))-1) + phi*Sfun(theta*heave,beta)*(1+theta*heave)
    Iv = velocity*np.cumsum((((1-phi*Sfun(theta,beta))**2)/kfun(theta,alpha))*dy*heave)
    return EffP*heave-IG-Ipsdt+Iv

@jit(nopython=True)
def forwardeuler(ye,dt,Nt,dy,N,Pe,porosity,Q,S,alpha,beta,G,nu,y,EffP,Ks,Ki,Kw,K):
    for i in range(Nt):
        theta = enthalpy2temperature(ye,N,porosity,S,beta)
        velocity = Pe*Vintegrals(theta,theta>=0,alpha,beta,G,nu,porosity,y,dy,EffP)
        if velocity > 0:
            afp = ye # advective flux +
            afm = np.append(ye[0],ye[:-1]) # advective flux -
        elif velocity < 0:
            afp = np.append(ye[1:],ye[-1]) # advective flux +
            afm = ye # advective flux -
        else:
            afp = np.zeros(N) # advective flux +
            afm = np.zeros(N) # advective flux -
        
        Kc = Kfun(theta,Ks,Ki,Kw,K,porosity,beta,alpha) # center
        Kp = np.append(Kc[1:],Kc[-1]) # positive
        Km = np.append(Kc[0],Kc[:-1]) # minus
        thetap = np.append(theta[1:],theta[-1])
        thetam = np.append(theta[0],theta[:-1])
        dfp = (2/dy)*((Kp*Kc)/(Kp+Kc))*(thetap-theta)
        dfm = (2/dy)*((Kc*Km)/(Kc+Km))*(theta-thetam)
        heaving = ye<=porosity
        fp = velocity*afp*heaving+dfp
        fm = velocity*afm*heaving+dfm
        fm[0]=1
        fp[-1]= Q
        ye = ye-dt*((fp-fm)/dy)
        if ~bool(i%(6e6)):
            print(i/Nt)
    return ye
        
if __name__ == "__main__":
    mesh = meshinformation()
    prms = parameters()
    bracket = scales()
    nd = nondimensional()
    # # timestep # #
    D = ((prms.Ks**(1-prms.phi))*(prms.Ki**prms.phi))*nd.S/prms.phi # diffusivity
    dt = (1/(2.01*D))*(mesh.dy**2) # timestep
    Nt = int(np.round(mesh.TotalTime/dt)) # number of timesteps
    
    # # initial temperature profile # #
    enthalpy = 0.36-0.04*mesh.y 
    start = time.time()
    esol = forwardeuler(enthalpy,dt,Nt,mesh.dy,mesh.N,nd.Pe,prms.phi,nd.Qout,nd.S,prms.alpha,prms.beta,nd.G,nd.nu,mesh.y,nd.EffP,prms.Ks,prms.Ki,prms.Kw,bracket.K)
    np.savetxt('esol_varyK.txt', (mesh.y,esol))
    end = time.time()
    print("Elapsed FE (with compilation) = %s" % (end - start))
   
    f = plt.figure()
    plt.plot(esol/prms.phi,mesh.y,'k.-')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylim(0,1)
    plt.xlim(0.75,1.005)
    plt.ylabel('height, $z$')
    plt.xlabel('enthalpy, $H/\phi$')
    
    theta = enthalpy2temperature(esol,mesh.N,prms.phi,nd.S,prms.beta)
    V = Vintegrals(theta,theta>=0,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP)
    print(100*abs((V-nd.V)/nd.V)) # percent error in V
    
    def ffintegral(zf): 
        tp = lambda t,x: (nd.Qout + ((1-nd.Qout)/((1+x)**prms.beta)))/Kfun(x,prms.Ks,prms.Ki,prms.Kw,bracket.K,prms.phi,prms.beta,prms.alpha)
        sol = solve_ivp(tp,[zf,1],[0],method='LSODA',dense_output=True)
        theta = lambda z: sol.sol(z)
        It1 = nd.G*(nd.nu-1)*(1-prms.phi)*(1-zf)
        It2 = (1-prms.phi)*theta(1) + (prms.phi/(1-prms.beta))*(((1+theta(1))**(1-prms.beta))-1)
        Ib = quad(lambda x: ((1-prms.phi*Sfun(theta(x),prms.beta))**2)/kfun(theta(x),prms.alpha),zf,1)
        return (1-nd.EffP+It1+It2)/Ib[0]
    
    zf = fsolve(lambda x: ffintegral(x)-nd.V,0.48)
    tp = lambda t,x: (nd.Qout + ((1-nd.Qout)/((1+x)**prms.beta)))/Kfun(x,prms.Ks,prms.Ki,prms.Kw,bracket.K,prms.phi,prms.beta,prms.alpha)
    sol = solve_ivp(tp,[zf,1],[0],method='LSODA',dense_output=True)
    esol_analytical = prms.phi*((1+sol.y.T)**(-prms.beta))
    plt.plot(esol_analytical/prms.phi,sol.t,'r')
    z_lower = np.linspace(0,zf,100)
    esol_lower = -(prms.phi/nd.S)*(z_lower-zf)+prms.phi
    plt.plot(esol_lower/prms.phi,z_lower,'r')
    f.savefig("enthalpyheight_varyK.pdf")
    plt.show()
    
    Nloc = EffPloc(theta,theta>=0,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP,V)
    
    f = plt.figure()
    plt.plot(Nloc/nd.EffP,mesh.y,'k.-')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.ylabel('height, $z$')
    plt.xlabel('scaled local effective pressure, $N_{\\textrm{loc}}/N$')
    f.savefig("localEffP_varyK.pdf")
    plt.show()