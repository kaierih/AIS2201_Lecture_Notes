import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt 
from numpy import cos, sin, pi, tan


def zp2tf(zeroes = np.array([]), poles = np.array([]), w_ref = 0, gain = 1):
    if zeroes.any()==False:
        b = np.array([1.0])
    else:
        b = np.poly(zeroes)
    if poles.any()==False:
        a = np.array([1.0])
    else: 
        a = np.poly(poles)
    # Find frequency response at w=w_ref
    H_ref = sig.freqz(b, a, worN=[w_ref])[1][0]
    b = b*(gain/np.abs(H_ref))
    return b, a

def tfPlot(b, a, ax=None):
    if ax == None:
        ax = plt.axes(projection='3d')
    res=122
    x = np.linspace(-1.2, 1.2, res)
    y = np.linspace(-1.2, 1.2, res)
    x,y = np.meshgrid(x,y)
    z = x + 1j*y
    Bz = np.zeros((res, res))*1j
    for i in range(len(b)):
        Bz += b[i]/(z**i)
    Az = np.zeros((res, res))*1j
    for i in range(len(a)):
        Az += a[i]/(z**i)
    Hz = Bz/Az
    
    ax.set_xlabel(r'$\mathcal{Re}(z)$')
    ax.set_xlim(-1.2,1.2)
    ax.set_ylabel(r'$\mathcal{Im}(z)$')
    ax.set_ylim(-1.2,1.2)
    ax.set_zlabel(r'$\left| H(z) \right|$ (dB)')

    plt.title(r'Visualisering av transferfunksjon $H(z)$')
    
    ax.plot_surface(x, y, 20*np.log10(np.abs(Hz)), rstride=1, cstride=1, cmap='viridis', edgecolor='none' )
    w, Hw = sig.freqz(b, a, worN=509, whole=True)
    x_w = np.cos(w)
    y_w = np.sin(w)
    ax.plot(x_w, y_w, 20*np.log10(np.abs(Hw)), linewidth=3, color='tab:red')
    plt.tight_layout()
    
def pzPlot(b, a):
    zeroes, poles, k = sig.tf2zpk(b, a)
    
    zpDiff = len(zeroes) - len(poles)
    
    if (zpDiff > 0):
        poles = np.concatenate((poles, np.zeros(np.abs(zpDiff))))
    elif (zpDiff < 0):
        zeroes = np.concatenate((zeroes, np.zeros(np.abs(zpDiff))))                        

    plt.plot(np.real(zeroes), np.imag(zeroes),'C0o', markersize=8, linewidth=0, markerfacecolor='none')
    plt.plot(np.real(poles), np.imag(poles),'C0x', markersize=8, linewidth=0, markerfacecolor='none')

    plt.grid(True)
    plt.axis([-1.2, 1.2, -1.2, 1.2])
    plt.plot(np.cos(np.linspace(0, 2*np.pi, 513)), np.sin(np.linspace(0, 2*np.pi, 513)),'C3:')
    plt.title('Pole zero map')
    plt.xlabel(r'$\mathcal{Re}(z)$')
    plt.ylabel(r'$\mathcal{Im}(z)$')
                               
    if (not (-1<= zpDiff <=1)):
        plt.annotate(np.abs(zpDiff), (0.02, 0.02), xytext=(0.0, 0.0),textcoords='offset points', xycoords='data', size='large')


                               
                               
def Magnitude_dB(b, a):
    w, Hw = sig.freqz(b, a, worN=509, whole=True)

    plt.plot(w, 20*np.log10(np.abs(Hw)))
    plt.grid(True)
    plt.ylim(ymin=-60)
    plt.xlim([0, np.pi])
    plt.title('Magnitude Response')
    plt.xticks(np.linspace(0, 1, 5)*np.pi, [r'$'+str(round(i,2))+'\pi$' for i in np.linspace(0, 1, 5)])
    plt.xlabel(r'Digital Frequency $\hat{\omega}$')
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
def visualizeTF(b, a ,fig_num=1):
    plt.close(fig_num)
    fig = plt.figure(fig_num, figsize = (8,6))
    
    ax = plt.subplot(2,3,(2,6),projection = '3d')
    tfPlot(b, a, ax)

    plt.subplot(2,3,1)    
    pzPlot(b, a)
    
    plt.subplot(2,3,4)   
    Magnitude_dB(b, a)

# Funksjon for å visualisere funksjonen e^-st
def showOscillation(s, T=2, fig_num=1, figsize=(9,6)):
    plt.close(fig_num);plt.figure(fig_num, figsize=(9,6))
    ax = plt.axes(projection='3d')

    t = np.linspace(0, T, 1001)
    signal = np.exp(s*t)
    
    yline = np.real(signal)
    zline = np.imag(signal)
    
    ax.plot3D(t, yline, zline)
    ax.set_xlabel('Tid (sekund)')
    ax.set_ylabel(r'Reell akse')
    ax.set_zlabel(r'Imaginær akse')
    ax.set_title(r"""3D-graf av: $e^{%s\cdot t}\cdot e^{%s\cdot t} $""" 
                 % (str(1*(np.real(s))).strip('()'),
                    str(1j*np.imag(s)).strip('()'))
                )
    plt.tight_layout()
    
# Funksjon for å visualisere funksjonen z^-n    
def showDiscreteOscillation(z, N=32, fig_num=1, figsize=(9,6)):
    plt.close(fig_num);plt.figure(fig_num, figsize=(9,6))
    ax = plt.axes(projection='3d')

    n = np.arange(N)
    t = np.linspace(0, N, 1001)
    z_n = z**n
    z_t = z**t
    yline = np.real(z_t)
    zline = np.imag(z_t)
    ydots = np.real(z_n)
    zdots = np.imag(z_n)
    
    ax.plot3D(t, yline, zline, ':')
    ax.scatter(n, ydots, zdots)
    ax.set_xlabel('Samplenummer')
    ax.set_ylabel(r'Reell akse')
    ax.set_zlabel(r'Imaginær akse')
    ax.set_title(r"""3D-graf av: $z^{n} = %s ^{n}\cdot e^{%s\pi \cdot n} $""" 
                 % (str(1*(np.abs(z))).strip('()'),
                    str(1j*round(np.angle(z)/np.pi,3)).strip('()'))
                )
    plt.tight_layout()

# Visualisering av en transferfunksjon H(s) i s-planet
def HsPlot(b, a, ax=None, axes=[-4, 4, -6, 6]):
    if ax == None:
        ax = plt.axes(projection='3d')
    res=122
    x = np.linspace(axes[0], axes[1], res)
    y = np.linspace(axes[2], axes[3], res)
    w = np.linspace(axes[2], axes[3], res*3)

    x,y = np.meshgrid(x,y)
    s = x + 1j*y
    Bs = np.zeros((res, res))*1j
    Bw = np.zeros(res*3)*1j
    for i in range(len(b)):
        Bs += b[i]*(s**(len(b)-i))
        Bw += b[i]*((1j*w)**(len(b)-i))
    Aw = np.zeros(res*3)*1j
    As = np.zeros((res, res))*1j
    for i in range(len(a)):
        As += a[i]*(s**(len(a)-i))
        Aw += a[i]*((1j*w)**(len(a)-i))
    Hs = Bs/As
    Hw = Bw/Aw
    
    ax.set_xlabel(r'$\sigma$')
    ax.set_xlim([axes[0], axes[1]])
    ax.set_ylabel(r'$j\omega$')
    ax.set_ylim([axes[2], axes[3]])
    ax.set_zlabel(r'$\left| H(s) \right|$ (dB)')

    plt.title(r'Visualisering av transferfunksjon $H(s)$')
    
    ax.plot_surface(x, y, 20*np.log10(np.abs(Hs)), rstride=1, cstride=1, cmap='viridis', edgecolor='none' )
    ax.plot(np.zeros(res*3), w, 20*np.log10(np.abs(Hw)), linewidth=3, color='tab:red')
    plt.tight_layout()
    
def displayFrequencyResponse(b, a=[1], mag='log', label=None):
    w, Hw = sig.freqz(b, a)
    H_amp = np.abs(Hw)
    H_phase = np.unwrap(np.angle(Hw))
    plt.subplot(2,1,1)
    if mag.lower()=='log':
        plt.plot(w, 20*np.log10(H_amp), label=label)
        plt.ylabel(r'$\left| H\left(\hat{\omega}\right)\right|$ (dB)')
    else:
        plt.plot(w, H_amp, label=label)
        plt.ylabel(r'$\left| H\left(\hat{\omega}\right)\right|$')
    plt.grid(True)
    plt.xticks(np.linspace(0, 1, 6)*pi, [str(round(i,2))+r'$\pi$' for i in np.linspace(0, 1, 6)])
    plt.xlabel(r'Digital Frekvens $\hat{\omega}$')

    plt.xlim([0, pi])
    plt.title(r'Frekvensrespons $H\left(\hat{\omega}\right)$')
    plt.legend()
    ax_phase = plt.subplot(2,1,2)
    phaseResp, = plt.plot(w, H_phase/pi, label=label)
    yticks = ax_phase.get_yticks()
    ylim = ax_phase.get_ylim()
    plt.grid(True)
    plt.xticks(np.linspace(0, 1, 6)*pi, [str(round(i,2))+r'$\pi$' for i in np.linspace(0, 1, 6)])
    plt.yticks(yticks, [str(round(i,2))+r'$\pi$' for i in yticks])
    plt.xlabel(r'Digital Frekvens $\hat{\omega}$')
    plt.ylabel(r'$\angle H\left(\hat{\omega}\right)$')
    plt.xlim([0, pi])
    plt.ylim(np.array(ylim))
    plt.legend()
    plt.tight_layout(pad=0.1, w_pad=1.0, h_pad=1.0)
    
    