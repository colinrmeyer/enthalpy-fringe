from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad, trapz
from scipy.optimize import fsolve
import time

class meshinformation():
    def __init__(self):
        # # mesh information # #
        # domain from from z=l to z=r
        self.l = 0 # nondimensional bottom
        self.r = 7 # nondimensional top
        self.N = (2**5)*(self.r-self.l) # Number of grid cells
        self.dy = (self.r-self.l)/self.N # grid cell width
        self.yedges = np.linspace(self.l,self.r,self.N+1) # cell edges
        self.y = (self.yedges[:self.N]+self.yedges[-self.N:])/2 # cell centers
        # # time information # #
        self.TotalTime = 100 # total nondimensional time

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
        self.Ks = 4.0; # sediment thermal conductivity, W/(m*degC)
        self.Latent = 3.34e5; # latent heat of fusion, J/kg
        self.g = 9.806; # acceleration due to gravity, m/s^2
        self.qin = 70e-3; # geothermal heat flux, 60 mW/m^2
        self.Tm = 273.15; # melting temperature, Kelvin
        self.k0 = 1e-17; # permeability, m^2
        self.gamma = 0.034; # interfacial energy, J/m^2
        self.mu = 1.8e-3; # viscosity of water, Pa*s
        self.rp = 1e-6; # grain size, m
        self.alpha = 3.1; # exponent in permeability as a function of temperature
        self.beta = 0.53; # exponent in saturation as a function of temperature
        self.EffP = effpval; # effective pressure, Pa
        self.sigman = 1e6; # overburden pressure, Pa
        self.phi = 0.35; # porosity
        self.y2s = 3600*365*24; # 1 year in seconds, s
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
    # Ib = np.trapz(y*heave,(((1-phi*Sfun(theta,beta))**2)/kfun(theta,alpha))*heave)
    return (1-EffP+It1+It2)/Ib

@jit(nopython=True)
def EffPloc(theta,heave,alpha,beta,G,nu,phi,y,dy,EffP,velocity):
    IG = G*np.cumsum((nu*(1-phi)+phi*(1-Sfun(theta,beta)))*dy*heave)
    Ipsdt = phi*theta*heave + (phi/(1-beta))*(((1+theta*heave)**(1-beta))-1) + phi*Sfun(theta*heave,beta)*(1+theta*heave)
    Iv = velocity*np.cumsum((((1-phi*Sfun(theta,beta))**2)/kfun(theta,alpha))*dy*heave)
    return EffP*heave-IG-Ipsdt+Iv

# # new lens event # #
@jit(nopython=True)
def lensfun(t,y,N,Stefan,alpha,beta,G,nu,phi,z,dz,EffP): 
    theta = enthalpy2temperature(y,N,phi,Stefan,beta)
    heaving = theta>=0
    V = Vintegrals(theta,heaving,alpha,beta,G,nu,phi,z,dz,EffP)
    Nloc = EffPloc(theta,heaving,alpha,beta,G,nu,phi,z,dz,EffP,V)
    return min(Nloc[heaving])

@jit(nopython=True)
def enthalpyode(time,ye,N,y,dy,phi,Stefan,alpha,beta,G,nu,EffP,Pe,V,Ks,Ki,Kw,K):
        theta = enthalpy2temperature(ye,N,phi,Stefan,beta)
        # impose force balance velocity
        heaving = ye<=0
        velocity_fb = Pe*Vintegrals(theta,heaving,alpha,beta,G,nu,phi,y,dy,EffP)
        if velocity_fb > 0:
            afp = velocity_fb*heaving*ye # advective flux +
            # afm = velocity_fb*np.append(heaving[0],heaving[:-1])*np.append(ye[0],ye[:-1]) # advective flux -
            afm = velocity_fb*heaving*np.append(ye[0],ye[:-1]) # advective flux -
        elif velocity_fb < 0:
            # afp = velocity_fb*np.append(heaving[1:],heaving[-1])*np.append(ye[1:],ye[-1]) # advective flux +
            afp = velocity_fb*heaving*np.append(ye[1:],ye[-1]) # advective flux +
            afm = velocity_fb*heaving*ye # advective flux -
        else:
            afp = np.zeros(N) # advective flux +
            afm = np.zeros(N) # advective flux -
        
        Kc = Kfun(theta,Ks,Ki,Kw,K,phi,beta,alpha) # center
        Kp = np.append(Kc[1:],Kc[-1]) # positive
        Km = np.append(Kc[0],Kc[:-1]) # minus
        thetap = np.append(theta[1:],theta[-1])
        thetam = np.append(theta[0],theta[:-1])
        dfp = (2/dy)*((Kp*Kc)/(Kp+Kc))*(thetap-theta)
        dfm = (2/dy)*((Kc*Km)/(Kc+Km))*(theta-thetam)
        dfm[0]= 1
        dfp[-1]= 1 - Pe*V*ye[-1]
        fp = afp+dfp
        fm = afm+dfm
        return -((fp-fm)/dy)
        
if __name__ == "__main__":
    effectivepressure = 200*1000 # Pa 
    
    mesh = meshinformation()
    prms = parameters(effectivepressure)
    bracket = scales(effectivepressure)
    
    # Vheaverate = -0.05*prms.qin/(prms.rhoi*prms.Latent) # m/s
    Vheaverate = 0.05*bracket.V # m/s
    # Vheaverate = 0.0 # m/s
    nd = nondimensional(effectivepressure,Vheaverate)
    
    # # lowest ice lens height # #
    zl = mesh.r # top of domain
    
    # # initial temperature profile # #
    thetai = mesh.y-4
    thetalensi = 10
    enthalpyi = temperature2enthalpy(thetai,thetalensi,mesh.N,prms.phi,nd.Stefan,prms.beta)
    
    numlens = 5
    for i in range(numlens):
        # # event function # #
        new_lens = lambda t,y: lensfun(t,y,mesh.N,nd.Stefan,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP)
        new_lens.terminal = True
        new_lens.direction = -1
        
        # # ode solution # #
        sol = solve_ivp(lambda t,x: enthalpyode(t,x,mesh.N,mesh.y,mesh.dy,prms.phi,nd.Stefan,prms.alpha,prms.beta,nd.G,nd.nu,nd.EffP,nd.Pe,nd.V,prms.Ks,prms.Ki,prms.Kw,bracket.K),[0,mesh.TotalTime],enthalpyi,method='LSODA',events=new_lens)
        
        # # find local effective pressure ## 
        theta = enthalpy2temperature(sol.y[:,-1],mesh.N,prms.phi,nd.Stefan,prms.beta)
        heaving = theta>=0
        V = Vintegrals(theta,heaving,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP)
        Nloc = EffPloc(theta,heaving,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP,V)
        
        # # collect new initial condition # #
        enthalpyi = np.append(sol.y[1,-1]*np.ones(np.size(sol.y[mesh.y>mesh.y[(min(Nloc[heaving])==Nloc)*heaving],-1])),sol.y[mesh.y<=mesh.y[(min(Nloc[heaving])==Nloc)*heaving],-1])
        
        f = plt.figure()
        plt.plot(Nloc/nd.EffP,mesh.y,'k.-')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.ylim(mesh.l,mesh.r)
        plt.xlim(0,1)
        plt.ylabel('height, $z$')
        plt.xlabel('scaled local effective pressure, $N_{\\textrm{loc}}/N$')
        f.savefig("localEffP_newlens%d.pdf"%(i+1))
        plt.show()
        
        print(i)
        print(sol.t_events)
        
    f = plt.figure()
    plt.plot(sol.y[:,-1],mesh.y,'k.-')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylim(mesh.l,mesh.r)
    plt.xlim(-0.3,0.02)
    plt.ylabel('height, $z$')
    plt.xlabel('enthalpy, $H$')
    
    theta = enthalpy2temperature(sol.y[:,-1],mesh.N,prms.phi,nd.Stefan,prms.beta)
    heaving = theta>=0
    V = Vintegrals(theta,heaving,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP)
    if nd.V==0:
        print(V) # V value
    else:
        print(100*(V-nd.V)/nd.V) # dV percentage
      
    f2 = plt.figure()
    plt.plot(theta,mesh.y,'k.-')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylim(mesh.l,mesh.r)
    plt.xlim(-3,5)
    plt.yticks(size=14)
    plt.xticks(size=14)
    plt.ylabel(r'height, $z$', fontsize=16)
    plt.xlabel(r'temperature, $\theta$', fontsize=16)
    
    Nloc = EffPloc(theta,heaving,prms.alpha,prms.beta,nd.G,nd.nu,prms.phi,mesh.y,mesh.dy,nd.EffP,V)
    f = plt.figure()
    plt.plot(Nloc/nd.EffP,mesh.y,'k.-')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.ylim(mesh.l,mesh.r)
    plt.xlim(0,1)
    plt.ylabel('height, $z$')
    plt.xlabel('scaled local effective pressure, $N_{\\textrm{loc}}/N$')
    # f.savefig("localEffP_newlens.pdf")
    plt.show()
    
    IG = nd.G*np.cumsum((nd.nu*(1-prms.phi)+prms.phi*(1-Sfun(theta,prms.beta)))*mesh.dy*heaving)
    Ipsdt = prms.phi*theta*heaving + (prms.phi/(1-prms.beta))*(((1+theta*heaving)**(1-prms.beta))-1) + prms.phi*Sfun(theta*heaving,prms.beta)*(1+theta*heaving)
    Iv = V*np.cumsum((((1-prms.phi*Sfun(theta,prms.beta))**2)/kfun(theta,prms.alpha))*mesh.dy*heaving)
    Nloc_out = nd.EffP*heaving-IG-Ipsdt+Iv
    
    print(sol.t_events)