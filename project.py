import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def difco(t,p,zf):
        
        k = np.zeros(len(zf)) 
        zmixt = p.zMix+(1-np.sin((t-91)*2*np.pi/365))*(p.zMixW-p.zMix)/2 
        k = 0.5*(1-np.tanh(((zf-zmixt)/p.zetaMix)))*(p.kTop-p.kBot)+p.kBot   

        return k


def fluxes(phi, p, t, zf, v, k, dz):
    
    Ja = np.zeros(len(zf))
    Jd = np.zeros(len(zf))

    Ja[1:-1] = v*phi[:len(phi)-1]
                    
    Jd[1:-1] = -k[1:-1]*(phi[1:]-phi[:-1])/dz
        
    return (Ja+Jd)


def olf(o):                                             # oxygen limiting factor     checked
    return 0.5*(1+np.tanh((o-100)/10))


def light(t,z,P,D,p):
    
    I_sur = (p.I0/2+2)+np.sin(2*np.pi*(t-91)/365)*p.I0/2
    I = I_sur*np.exp(-p.kw*z-np.cumsum(P+D)*p.dzw*p.kp)

    return I

def mainf(dzw, dzs, tend):
    
    class parameters:
        def __init__(self):


            # water column & Sediment
            self.zw = 15                        # depth of water column                          (m)
            self.zs = 1                         # depth of sediment layer                        (m)
            self.zbot = self.zw + self.zs       # total depth                                    (m)
            self.dzw = dzw                     # thickness of grid cells in water column        (m)
            self.dzs = dzs                     # thickness of grid cells in sediment            (m)
            self.nw = int(self.zw/self.dzw)     # number of cells in water column                (-)
            self.ns = int(self.zs/self.dzs)     # number of cells in sediment                    (-)
            self.n = self.nw + self.ns          # number of cells                                (-)
            self.kTop = 5e-5*3600*24            # diffusivity coefficient at top                 (m^2/day)
            self.kBot = 5e-6*3600*24            # diffusivity coefficient at bottom              (m^2/day)
            self.diff_sed = 1e-7*3600*24        # diffusivity coefficient in sediment            (m^2/day)
            self.tstart = 0                     # start time of simulation                       (day)
            self.tend = tend                      # end of simulation                              (day)
            self.kw = 0.2                       # light attenuation by water                     (1/m)  
            self.kp = 0.05                      # light attenuation by phytoplankton             (1/(m.mmol N/m^3))
            self.kL = 4                       # light half saturation                           (mu mol photon/(m^2.s))
            self.zMix = self.zw/2             # depth at which diffusivity falls                (m)
            self.zetaMix = 1                  # thickness of stratification boundary            (m)
            self.zMixW = 0.9*self.zw         # zMix at winter 
            self.I0 = 70                      # surface light intensity                         (photons/(m^2.day))


            # phytoplankton
            self.gPmax = 1.5                   # max growth rate                                 (1/day)
            self.mp = 0.1                      # mortality rate                                  (1/day)
            self.kz = 2.0                      # phytoplankton half saturation                   (mmol N/m^3)
            self.P0 = 5                        # initial phytoplankton concentration             (mmol N/m^3)

            # Zooplankton
            self.gzmax = 0.7                   # zooplankton max grazing rate                    (1/day)
            self.e_res = 0.3                   # respiration fraction of zooplankton grazing     (-)
            self.e_eg = 0.3                    # egestion fraction of zooplankton grazing        (-)
            self.mz = 0.1                      # zooplankton mortality rate                      (1/day)
            self.Z0 = 2                        # initial zooplankton concentration              (mmol N/m^3)
            
            # Detritus
            self.wD = 5                        # sinking velocity                                (m/day)
            self.O2_lim = 5                    # oxygen threshold for aerobic mineralization     (mmol O2/m^3)
            self.D0 = 1                        # particulate or dissolved labile                 (mmol N/m^3)

            # Nitrate & Ammonium
            self.kN = 0.5                      # nutrient half saturation                        (mmol N/m^3)
            self.NH4_lim = 0.1                 # ammonium -> nitrate threshold for photosynthesis(mmol N/m^3)
            self.NO3_input = 0.5               # nitrate input at sea surface                    (mmol N/m^2/day)                  
            self.r_nit = 0.1                   # nitrification rate                              (1/day)
            self.r_denit = 0.2                 # denitrification rate                            (1/day)
            self.k_NO3 = 1                     # nitrate half saturation for denitrification     (mmol N/m^3)
            self.NH40 = 1                      # initial ammonium concentration                 (mmol N/m^3)
            self.NO30 = 3                      # initial nitrate concentration                 (mmol N/m^3)

            # Oxygen
            self.o2_atm = 260                   # atmospheric oxtgen                              (mmol O2/m^3)                                                     
            self.gtv = 1e-4*3600*24             # gas transfer velocity at sea surfaec            (m/day)                                               # ?????????????
            self.k_o = 10                       # oxygen half saturation                          (mmol O2/m^3)
            self.r_o2 = 0.1                     # rate of aerobic mineralization                  (1/day)
            self.O20 = 200                      # initial oxygen concentration                    (mmol O2/m^3)

    p=parameters()

    zw = np.linspace(p.dzw/2, p.zw - p.dzw/2, p.nw)                                         # water column cell centers depth
    zs = np.linspace(p.zw + p.dzs/2, p.zbot - p.dzs/2, p.ns)                                # sediment cell centers depth             

    zfw = np.linspace(0, p.zw , p.nw + 1)                                                # water column cell faces depth
    zfs = np.linspace(p.zw, p.zbot , p.ns +1)                                            # sediment cell faces depth

    def derivatives(t, y, p, zfw, zfs):
        
        P    = y[:p.nw]
        Z    = y[p.nw:2*p.nw]
        NH4w = y[2*p.nw:3*p.nw]
        NH4s = y[3*p.nw:3*p.nw+p.ns]
        NO3w = y[3*p.nw+p.ns:4*p.nw+p.ns]
        NO3s = y[4*p.nw+p.ns:4*p.nw+2*p.ns]
        Dw   = y[4*p.nw+2*p.ns:5*p.nw+2*p.ns]
        Ds   = y[5*p.nw+2*p.ns:5*p.nw+3*p.ns]
        Ow   = y[5*p.nw+3*p.ns:6*p.nw+3*p.ns]
        Os   = y[6*p.nw+3*p.ns:6*p.nw+4*p.ns]

        kw = difco(t, p, zfw)
        ks = np.ones(len(zfs)) * p.diff_sed

        JP    = fluxes(P,    p, t, zfw, 0,    kw, p.dzw)
        JZ    = fluxes(Z,    p, t, zfw, 0,    kw, p.dzw)
        JNH4w = fluxes(NH4w, p, t, zfw, 0,    kw, p.dzw)
        JNH4s = fluxes(NH4s, p, t, zfs, 0,    ks, p.dzs)
        JNO3w = fluxes(NO3w, p, t, zfw, 0,    kw, p.dzw)
        JNO3s = fluxes(NO3s, p, t, zfs, 0,    ks, p.dzs)
        JDw   = fluxes(Dw,   p, t, zfw, p.wD, kw, p.dzw)
        JDs   = fluxes(Ds,   p, t, zfs, 0,    ks, p.dzs)
        JOw   = fluxes(Ow,   p, t, zfw, 0,    kw, p.dzw)
        JOs   = fluxes(Os,   p, t, zfs, 0,    ks, p.dzs)

        # Boundary conditions
        
        JNH4w[-1] = -p.kBot * (NH4s[0] - NH4w[-1]) / (p.dzw/2 + p.dzs/2)
        JNH4s[0] = -p.kBot * (NH4s[0] - NH4w[-1]) / (p.dzw/2 + p.dzs/2)

        JNO3w[0]  = p.NO3_input
        JNO3w[-1] = -p.kBot * (NO3s[0] - NO3w[-1]) / (p.dzw/2 + p.dzs/2)
        JNO3s[0] = -p.kBot * (NO3s[0] - NO3w[-1]) / (p.dzw/2 + p.dzs/2)

        JDw[-1] = -p.kBot * (Ds[0] - Dw[-1]) / (p.dzw/2 + p.dzs/2)
        JDs[0] = -p.kBot * (Ds[0] - Dw[-1]) / (p.dzw/2 + p.dzs/2)

        JOw[0]   = -p.gtv * (Ow[0] - p.o2_atm)
        JOw[-1]  = -p.kBot * (Os[0] - Ow[-1]) / (p.dzw/2 + p.dzs/2)
        JOs[0] = -p.kBot * (Os[0] - Ow[-1]) / (p.dzw/2 + p.dzs/2)

        l = light(t, zw, P, Dw, p)

        # --- Water column 
        gz      = p.gzmax * P / (P + p.kz)
        olf_Ow  = olf(Ow)
        o_lim_w = Ow / (Ow + p.k_o)
        nit_w   = p.r_nit * NH4w * o_lim_w
        denit_w = p.r_denit * Dw * (NO3w / (NO3w + p.k_NO3))

        aerobic = Ow  > p.O2_lim     # aerobic degradation when O2 is sufficient

        N_lim_NH4 = NH4w/(p.kN+NH4w)
        N_lim_NO3 = NO3w*np.exp(-2*NH4w)/(p.kN+NO3w)
        N_lim = N_lim_NH4 + N_lim_NO3   

        gP       = p.gPmax * np.minimum(N_lim, l/(l+p.kL))
        degrad_w = np.where(aerobic, p.r_o2*Dw*o_lim_w, denit_w)

        divJP    = (JP[1:]    - JP[:-1])    / p.dzw
        divJNH4w = (JNH4w[1:] - JNH4w[:-1]) / p.dzw
        divJNO3w = (JNO3w[1:] - JNO3w[:-1]) / p.dzw
        divJDw   = (JDw[1:]   - JDw[:-1])   / p.dzw
        divJOw   = (JOw[1:]   - JOw[:-1])   / p.dzw

        dPdt    = -divJP + gP*P - olf_Ow*gz*Z - p.mp*P

        dNH4wdt = (-divJNH4w
                - nit_w
                + olf_Ow*p.e_res*gz*Z
                + degrad_w
                - gP*P*N_lim_NH4/N_lim)
        

        dNO3wdt = (-divJNO3w
                + nit_w
                - gP*P*N_lim_NO3/N_lim
                - np.where(aerobic, 0.0, denit_w*84.8/16))  # denitrification sink
        
        dDwdt   = (-divJDw
                + p.mp*P + p.mz*Z
                + olf_Ow*p.e_eg*gz*Z
                - degrad_w)

        dOwdt   = (-divJOw
                + 138/16*gP*P
                - olf_Ow*106/16*p.e_res*gz*Z
                - 2*nit_w
                - np.where(aerobic, 106/16*p.r_o2*Dw*o_lim_w, 0.0))

        dZdt = -(JZ[1:]-JZ[:-1])/p.dzw + (1-p.e_res-p.e_eg)*olf_Ow*gz*Z - p.mz*Z

        # --- Sediment (vectorized, replaces for loop) ---
        aerobic_s = Os > p.O2_lim
        o_lim_s   = Os / (Os + p.k_o)
        nit_s     = p.r_nit * NH4s * o_lim_s
        denit_s   = p.r_denit * Ds * (NO3s / (NO3s + p.k_NO3))
        degrad_s  = np.where(aerobic_s, p.r_o2*Ds*o_lim_s, denit_s)

        divJNH4s = (JNH4s[1:] - JNH4s[:-1]) / p.dzs
        divJNO3s = (JNO3s[1:] - JNO3s[:-1]) / p.dzs
        divJDs   = (JDs[1:]   - JDs[:-1])   / p.dzs
        divJOs   = (JOs[1:]   - JOs[:-1])   / p.dzs

        dNH4sdt = -divJNH4s - nit_s + degrad_s
        dNO3sdt = -divJNO3s + nit_s - np.where(aerobic_s, 0.0, denit_s*84.8/16)
        dDsdt   = -divJDs - degrad_s
        dOsdt   = (-divJOs
                - 2*nit_s
                - np.where(aerobic_s, 106/16*p.r_o2*Ds*o_lim_s, 0.0))

        return np.concatenate((dPdt, dZdt, dNH4wdt, dNH4sdt, dNO3wdt, dNO3sdt, dDwdt, dDsdt, dOwdt, dOsdt))


    # Initial conditions
    P0 = p.P0*((zw>=0.1*p.zw) * (zw<=0.5*p.zw))
    Z0 = p.Z0*((zw>=0.3*p.zw) * (zw<=1*p.zw))
    NH4w0 = p.NH40*((zw>=0.0*p.zw) * (zw<=p.zw))
    NH4s0 = p.NH40*((zs>=0.0*p.zs) * (zs<=p.zs))
    NO3w0 = p.NO30*((zw>=0.0*p.zw) * (zw<=p.zw))
    NO3s0 = p.NO30*((zs>=0.0*p.zs) * (zs<=p.zs))
    Dw0 = p.D0*((zw>=0.4*p.zw) * (zw<=0.95*p.zw))
    Ds0 = p.D0*((zs>=0.1*p.zs) * (zs<=0.6*p.zs))
    Ow0 = p.O20*((zw>=0.25*p.zw) * (zw<=0.9*p.zw))
    Os0 = p.O20*((zs>=0.3*p.zs) * (zs<=0.8*p.zs))


    y0 = np.concatenate((P0, Z0, NH4w0, NH4s0, NO3w0, NO3s0, Dw0, Ds0, Ow0, Os0))

    sol = solve_ivp(derivatives, t_span=(p.tstart, p.tend), y0=y0, method='RK45',
                    args=(p, zfw, zfs), rtol=1e-3, t_eval=np.arange(p.tstart, p.tend+1, 1))

    t    = sol.t
    P    = sol.y[:p.nw,:]
    Z    = sol.y[p.nw:2*p.nw,:]
    NH4w = sol.y[2*p.nw:3*p.nw,:]
    NH4s = sol.y[3*p.nw:3*p.nw+p.ns,:]
    NO3w = sol.y[3*p.nw+p.ns:4*p.nw+p.ns,:]
    NO3s = sol.y[4*p.nw+p.ns:4*p.nw+2*p.ns,:]
    Dw   = sol.y[4*p.nw+2*p.ns:5*p.nw+2*p.ns,:]
    Ds   = sol.y[5*p.nw+2*p.ns:5*p.nw+3*p.ns,:]
    Ow   = sol.y[5*p.nw+3*p.ns:6*p.nw+3*p.ns,:]
    Os   = sol.y[6*p.nw+3*p.ns:6*p.nw+4*p.ns,:]

    L = np.zeros((len(zw),len(t)))

    for i in np.arange(0,len(t),1):
        
        L[:,i] = light(t[i],zw,P[:,i],Dw[:,i],p)

    return t, zw, zs, zfw, zfs, P, Z, NH4w, NH4s, NO3w, NO3s, Dw, Ds, Ow, Os, L, p


# check mesh dependency --------------------------------------------------------------------
import math

tend=50
dzw = np.array([3,1.5,0.75,0.375,0.125])
dzs = np.array([0.5,0.25,0.125,0.075,0.05])
Pend = np.zeros(len(dzw))
c = 0

for i, j in zip(dzw, dzs):

    P = mainf(i,j,tend)[5][:,-1]  
    if (15/i)%2==0:
        Pend[c]=(P[int(len(P)/2)-1]+P[int(len(P)/2)])/2
    else:
        Pend[c]=P[math.ceil(len(P)/2)-1]
    c += 1

plt.figure(figsize=(4,3),dpi=300)
plt.plot(dzw,Pend,'-o')
plt.xlabel('dzw (m)',fontsize=13)
plt.title('P(mmol N/m^3) at z=7.5 m',fontsize=11)
plt.ylabel('P(mmol N/m^3)',fontsize=10)


# 3-year Run ------------------------------------------------------------------------------------------------
t, zw, zs, zfw, zfs, P, Z, NH4w, NH4s, NO3w, NO3s, Dw, Ds, Ow, Os, L,p = mainf(dzw=0.5, dzs=0.1, tend=3*365)


# Calculate rate of Aerobic decomposition/Denitrifcation, nitrification

aerobic_decomp_w = np.where(Ow > p.O2_lim, p.r_o2 * Dw * (Ow / (Ow + p.k_o)), 0)
denitrification_w = np.where(Ow > p.O2_lim, 0, p.r_denit * Dw * (NO3w / (NO3w + p.k_NO3)))
nitrification_w = p.r_nit * NH4w * (Ow / (Ow + p.k_o))

aerobic_decomp_s = np.where(Os > p.O2_lim, p.r_o2 * Ds * (Os / (Os + p.k_o)), 0)
denitrification_s = np.where(Os > p.O2_lim, 0, p.r_denit * Ds * (NO3s / (NO3s + p.k_NO3)))
nitrification_s = p.r_nit * NH4s * (Os / (Os + p.k_o))



# Plots ----------------------------------------------------------------------------------------------

# Line Graphs

# Winter
# Plotting NH4, NO3, P, Z

plt.figure(figsize=(10,11),dpi=300)
plt.subplot(3,2,1)
plt.plot(NH4w[:,-1],zw,label='Ammonium NH4 (mmol N/m^3)')
plt.plot(NO3w[:,-1],zw,label='Nitrate NO3 (mmol N/m^3)')
plt.plot(P[:,-1],zw,label='Phytoplankton P (mmol N/m^3)')
plt.plot(Z[:,-1],zw,label='Zooplankton Z (mmol N/m^3)')
plt.xlim(right=8.5)
plt.ylabel('Depth z(m)')
#plt.title('Winter')
plt.gca().invert_yaxis()
plt.legend(fontsize=7)

# Plotting limiting factors on phytoplankton growth
plt.subplot(3,2,2)
plt.plot(L[:,-1]/(L[:,-1]+p.kL),zw,label='Light Limitting Factor')
N_lim    = np.where(NH4w<p.NH4_lim, NO3w/(p.kN+NO3w), NH4w/(p.kN+NH4w))
#LN=NO3w[:,-1]*np.exp(-2*NH4w[:,-1])/(p.kN+NO3w[:,-1])+(NH4w[:,-1]/(p.kN+NH4w[:,-1]))
plt.plot(N_lim[:,-1],zw,label='Nutrient Limitting Factor')
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend()

# Plotting D 
plt.subplot(3,2,3)
plt.plot(Dw[:,-1],zw,label='Detritus D (mmol N/m^3)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend()

# Plotting O
plt.subplot(3,2,4)
plt.plot(Ow[:,-1],zw,label='Oxygen O2(mmol O2/m^3)')
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend()

plt.subplot(3,2,5)
plt.plot(aerobic_decomp_w[:,-1],zw,label='Aerobic Decomposition (mmol N/(m^3.day))')
plt.plot(denitrification_w[:,-1],zw,label='Denitrification (mmol N/(m^3.day))')
plt.plot(nitrification_w[:,-1],zw,label='Nitrification (mmol N/(m^3.day))')
plt.xlim(right=3.25)
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)

plt.subplot(3,2,6)
plt.plot(difco(t[-1],p,zfw),zfw, label='Diffusivity (m^2/day)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

# Summer ----------------------------------------------------------------------------------------------------
#Plotting NH4, NO3, P, Z

plt.figure(figsize=(10,11),dpi=300)
plt.subplot(3,2,1)
plt.plot(NH4w[:,-182],zw,label='Ammonium NH4 (mmol N/m^3)')
plt.plot(NO3w[:,-182],zw,label='Nitrate NO3 (mmol N/m^3)')
plt.plot(P[:,-182],zw,label='Phytoplankton P (mmol N/m^3)')
plt.plot(Z[:,-182],zw,label='Zooplankton Z (mmol N/m^3)')
plt.ylabel('Depth z(m)')
#plt.title('Summer')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)

# Plotting limiting factors on phytoplankton growth
plt.subplot(3,2,2)
plt.plot(L[:,-182]/(L[:,-182]+p.kL),zw,label='Light Limitting Factor')
#LN = np.where(NH4w[:,-1] < p.NH4_lim, NO3w[:,-1]/(p.kN+NO3w[:,-1]), NH4w[:,-1]/(p.kN+NH4w[:,-1]))
LN=NO3w[:,-182]*np.exp(-2*NH4w[:,-182])/(p.kN+NO3w[:,-182])+(NH4w[:,-182]/(p.kN+NH4w[:,-182]))
plt.plot(LN,zw,label='Nutrient Limitting Factor')
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)

# Plotting D 
plt.subplot(3,2,3)
plt.plot(Dw[:,-182],zw,label='Detritus D (mmol N/m^3)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend()

# Plotting O
plt.subplot(3,2,4)
plt.plot(Ow[:,-182],zw,label='Oxygen O2(mmol O2/m^3)')
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend()

plt.subplot(3,2,5)
plt.plot(aerobic_decomp_w[:,-182],zw,label='Aerobic Decomposition (mmol N/(m^3.day))')
plt.plot(denitrification_w[:,-182],zw,label='Denitrification (mmol N/(m^3.day))')
plt.plot(nitrification_w[:,-182],zw,label='Nitrification (mmol N/(m^3.day))')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)

plt.subplot(3,2,6)
plt.plot(difco(t[-182],p,zfw),zfw, label='Diffusivity (m^2/day)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

# # Colormaps --------------------------------------------------------------------------------------------------------------------------

from datetime import datetime, timedelta

# starting date
start_date = datetime(2024, 12, 21)

# convert to actual dates
dates = [start_date + timedelta(days=int(d)) for d in t]

import matplotlib.dates as mdates

# Plotting Light Colormap
plt.figure(figsize=(12,14),dpi=300)
plt.subplot(5,2,1)
plt.pcolormesh(dates, zw, L, shading='auto', cmap='viridis')
plt.title('Light Intensity (mu mol photon/(m^2.s))')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Nutrient Colormap
plt.subplot(5,2,2)
plt.pcolormesh(dates, zw, NH4w, shading='auto', cmap='viridis')
plt.title('Ammonium (mmol N/m^3)')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Nutrient Colormap
plt.subplot(5,2,3)
plt.pcolormesh(dates, zw, NO3w, shading='auto', cmap='viridis')
plt.title('Nitrate (mmol N/m^3)')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

plt.subplot(5,2,4)
plt.pcolormesh(dates, zw, aerobic_decomp_w, shading='auto', cmap='viridis')
plt.title('Aerobic Decomposition (mmol N/(m^3.day))')
plt.colorbar()
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

plt.subplot(5,2,5)
plt.pcolormesh(dates, zw, denitrification_w, shading='auto', cmap='viridis')
plt.title('Denitrification (mmol N/(m^3.day))')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

plt.subplot(5,2,6)
plt.pcolormesh(dates, zw, nitrification_w, shading='auto', cmap='viridis')
plt.title('Nitrification (mmol N/(m^3.day))')
plt.colorbar()
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Phytoplankton Colormap
plt.subplot(5,2,7)
plt.pcolormesh(dates, zw, P, shading='auto', cmap='viridis')
plt.title('Phytoplankton (mmol N/m^3)')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Nutrient Colormap
plt.subplot(5,2,8)
plt.pcolormesh(dates, zw, Dw, shading='auto', cmap='viridis')
plt.title('Detritus (mmol N/m^3)')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Zooplankton Colormap
plt.subplot(5,2,9)
plt.pcolormesh(dates, zw, Z, shading='auto', cmap='viridis')
plt.title('Zooplankton (m mol N/m^3)')
plt.colorbar()
plt.xlabel('Time (day)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Oxygen Colormap
plt.subplot(5,2,10)
plt.pcolormesh(dates, zw, Ow, shading='auto', cmap='viridis')
plt.title('Oxygen (m mol O2/m^3)')
plt.colorbar()
plt.xlabel('Time t(day)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

plt.tight_layout(pad=2.0)
plt.show()



# Sediment Plots-------------------------------------------------------------------------------


# Line Graphs

# Plotting NH4, NO3, P, Z

plt.figure(figsize=(9,8),dpi=300)
plt.subplot(2,2,1)
plt.plot(NH4s[:,-1],zs,label='Ammonium (mmol N/m^3)')
plt.plot(NO3s[:,-1],zs,label=' Nitrate (mmol N/m^3)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend(fontsize=7)

# Plotting O
plt.subplot(2,2,2)
plt.plot(Os[:,-1],zs,label='Oxygen O2(mmol O2/m^3)')
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend()

# Plotting D 
plt.subplot(2,2,3)
plt.plot(Ds[:,-1],zs,label='Detritus D (mmol N/m^3)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend()


plt.subplot(2,2,4)
plt.plot(aerobic_decomp_s[:,-1],zs,label='Aerobic Decomposition (mmol N/(m^3.day))')
plt.plot(denitrification_s[:,-1],zs,label='Denitrification (mmol N/(m^3.day))')
plt.plot(nitrification_s[:,-1],zs,label='Nitrification (mmol N/(m^3.day))')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.xlim(right=3.5)
plt.legend(fontsize=8,loc='lower right')

plt.show()

# summer
plt.figure(figsize=(9,8),dpi=300)
plt.subplot(2,2,1)
plt.plot(NH4s[:,-182],zs,label='Ammonium (mmol N/m^3)')
plt.plot(NO3s[:,-182],zs,label=' Nitrate (mmol N/m^3)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)

# Plotting D 
plt.subplot(2,2,3)
plt.plot(Ds[:,-182],zs,label='Detritus D (mmol N/m^3)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)

# Plotting O
plt.subplot(2,2,2)
plt.plot(Os[:,-182],zs,label='Oxygen O2(mmol O2/m^3)')
#plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.legend(fontsize=9)

plt.subplot(2,2,4)
plt.plot(aerobic_decomp_s[:,-182],zs,label='Aerobic Decomposition (mmol N/(m^3.day))')
plt.plot(denitrification_s[:,-182],zs,label='Denitrification (mmol N/(m^3.day))')
plt.plot(nitrification_s[:,-182],zs,label='Nitrification (mmol N/(m^3.day))')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.xlim(right=6)
plt.legend(fontsize=8, loc='lower right')

plt.show()

# Colormaps --------------------------------------------------------------------------------------------------------------------------


from datetime import datetime, timedelta

# starting date
start_date = datetime(2024, 12, 21)

# convert to actual dates
dates = [start_date + timedelta(days=int(d)) for d in t]

import matplotlib.dates as mdates

# Plotting Light Colormap
plt.figure(figsize=(12,12),dpi=300)

# Plotting Ammonium Colormap
plt.subplot(4,2,1)
plt.pcolormesh(dates, zs, NH4s, shading='auto', cmap='viridis')
plt.title('Ammonium (mmol N/m^3)')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')


# Plotting Nitrate Colormap
plt.subplot(4,2,2)
plt.pcolormesh(dates, zs, NO3s, shading='auto', cmap='viridis')
plt.title('Nitrate (mmol N/m^3)')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

plt.subplot(4,2,3)
plt.pcolormesh(dates, zs, aerobic_decomp_s, shading='auto', cmap='viridis')
plt.title('Aerobic Decomposition (mmol N/(m^3.day))')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

plt.subplot(4,2,4)
plt.pcolormesh(dates, zs, denitrification_s, shading='auto', cmap='viridis')
plt.title('Denitrification (mmol N/(m^3.day))')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right') 

plt.subplot(4,2,5)
plt.pcolormesh(dates, zs, nitrification_s, shading='auto', cmap='viridis')
plt.title('Nitrification (mmol N/(m^3.day))')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Detritus Colormap
plt.subplot(4,2,6)
plt.pcolormesh(dates, zs, Ds, shading='auto', cmap='viridis')
plt.title('Detritus (mmol N/m^3)')
plt.colorbar()
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

# Plotting Oxygen Colormap
plt.subplot(4,2,7)
plt.pcolormesh(dates, zs, Os, shading='auto', cmap='viridis')
plt.title('Oxygen (m mol O2/m^3)')
plt.colorbar()
plt.xlabel('Time t(day)')
plt.ylabel('Depth z(m)')
plt.gca().invert_yaxis()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=30, ha='right')

plt.tight_layout(pad=2)

plt.show()

# Aerobic vs Anaerobic Decomposition & N2 Loss --------------------------------------------------------------------------

from datetime import datetime, timedelta

# starting date
start_date = datetime(2024, 12, 21)

# convert to actual dates
dates = [start_date + timedelta(days=int(d)) for d in t]

import matplotlib.dates as mdates

aerobic_s = Os > p.O2_lim
o_lim_s   = Os / (Os + p.k_o)
denit_s   = p.r_denit * Ds * (NO3s / (NO3s + p.k_NO3))
aer_s = np.where(aerobic_s, p.r_o2*Ds*o_lim_s, 0)
anaer_s  = np.where(aerobic_s, 0, denit_s)


aerobic = Ow  > p.O2_lim     
o_lim_w = Ow / (Ow + p.k_o)
denit_w = p.r_denit * Dw * (NO3w / (NO3w + p.k_NO3))
aer_w = np.where(aerobic, p.r_o2*Dw*o_lim_w, 0)
anaer_w = np.where(aerobic, 0, denit_w)


aer= aer_s.sum(axis=0)*p.dzs + aer_w.sum(axis=0)*p.dzw
anaer = anaer_s.sum(axis=0)*p.dzs + anaer_w.sum(axis=0)*p.dzw
N2_loss = anaer*42.4/16

# x-axis date range (adjust as needed)
# xlim_start = datetime(2026, 6, 21)
# xlim_end   = datetime(2026, 7, 21)

plt.figure(figsize=(7,5),dpi=300)
plt.plot(dates, aer/(aer+anaer)*100)
plt.title('% Aerobic Decomposition')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.xlim([xlim_start, xlim_end])
plt.xticks(rotation=30, ha='right')
plt.show()

plt.figure(figsize=(7,5),dpi=300)
plt.plot(dates, N2_loss)
plt.title('N2 Loss')
plt.ylabel('N2 (mmol N2/day)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.xlim([xlim_start, xlim_end])
plt.xticks(rotation=30, ha='right')
plt.show()
