# Created June 25, 2014
# To plot the LF of the real COSMOS data and guess the parameters of the Schechter Function

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import interpolate
from scipy.integrate import quad

import pymultinest
import pymultinest_schechterfit

#Get data file
dat5 = pyfits.getdata('../great3_fit_data/real_galaxy_catalog_23.5_fits.fits');
print "Data file 5 read ... \n";
dat0 = pyfits.getdata('../Data/lensing14.fits');
print "Data file read ... \n";

#Find the mapping between dat5 and dat0
ident0 = dat0.field('IDENT');
ident5 = dat5.field('IDENT');

ident = np.loadtxt('ident_mapping_5.txt').astype(int);

#Convert the Look-Up Table for D_A vs Z to a function of redshift
lut = np.loadtxt('da_vs_z.txt');
def da_definition(xp,fp):
	def f(z):
		return np.interp(z,xp,fp);
	return f;	
DA = da_definition(lut[:,0],lut[:,1]);

#Mcuts for binwise volume-limiting
Mcuts = np.loadtxt('bvl_Mcuts.txt',comments='#');

mag_auto = (dat0.field('MAG_AUTO'));
mi = (dat0.field('MI'));
zphot = (dat0.field('ZPHOT'));
r_0p5_gim2d = (dat0.field('R_0P5_GIM2D'));
clean = dat0.field('CLEAN');
good_zphot_source = dat0.field('GOOD_ZPHOT_SOURCE');
mu_class = dat0.field('MU_CLASS');

mag_auto0 = mag_auto[ident[:,2]];
mi0 = mi[ident[:,2]];
zphot0 = zphot[ident[:,2]];
r_0p5_gim2d0 = r_0p5_gim2d[ident[:,2]]
clean0 = clean[ident[:,2]];
good_zphot_source0 = good_zphot_source[ident[:,2]];
mu_class0 = mu_class[ident[:,2]];
#selection_index = ((zphot0>=z1a)&(zphot0<z1b)&(clean0==1)&(good_zphot_source0==1)&(mu_class0==1)).astype(bool);
superselection_index = ((clean==1)&(good_zphot_source==1)&(mu_class==1)).astype(bool);
selection_index = superselection_index[ident[:,2]];

#Cosmology
OmegaM = 0.3; OmegaK = 0; OmegaL = 0.7;

z1a = 0.3; z1b = 0.4;

#Mmax = -18.35; #for 0.3 - 0.4
Mmax = -20.0 #for 0.65-0.75
#Mmax = -20.9 #for 0.80 - 0.85

Mmax = np.max(mi0[selection_index&(zphot0>=z1a)&(zphot0<z1b)&(mi0<Mcuts[0,3])&(mi0>-25)]);

Z = zphot0[selection_index&(zphot0>=z1a)&(zphot0<z1b)&(mi0<Mcuts[0,3])&(mi0>-25)&(mi0<Mmax)];
EZ = np.sqrt(OmegaM*(1+Z)**3+OmegaK*(1+Z)**2+OmegaL);
invW = ((1+Z)**2)*DA(Z)/EZ;

M = mi0[selection_index&(zphot0>=z1a)&(zphot0<z1b)&(mi0<Mcuts[0,3])&(mi0>-25)&(mi0<Mmax)];
L = 10**(-0.4*M);

#defining Schechter function
Mag = np.linspace(-25,Mmax,1001);
#norm = 0.14; alpha = -1.25; Mstar = -22.8 #for 0.3 - 0.4. THe next line seems better.
norm = 0.2; alpha = -1.1; Mstar = -22.35 #for 0.3 - 0.4  
#norm = 0.5; alpha = -1.05; Mstar = -22.5 #for 0.65 - 0.75
#norm = 0.65; alpha = -1.05; Mstar = -22.6 #for 0.80 - 0.85

fig, ax = plt.subplots(2,2);   
fig.suptitle(r'ML Estimate of LF parameters for $z:$'+str(z1a)+' - '+str(z1b)+' with \n volume-limited sample size '+str(len(Z))+' & bin-wise-volume-limited sample size '+'\n'+r' 1$\sigma$ errorbars' );  
   
svl = pymultinest_schechterfit.vl_2p(len(Z),L.mean(),M.mean(),Mmax, figax = (fig,ax));
alpha = svl['marginals'][0]['median']; Mstar = svl['marginals'][1]['median'];
alpha_low, alpha_high = svl['marginals'][0]['1sigma']; Mstar_low, Mstar_high = svl['marginals'][1]['1sigma']

print "Modes: ", svl['modes']

print "Parameter alpha ~ ", alpha, "[ ", alpha_low, ",", alpha_high, "]";
print "Parameter Mstar ~ ", Mstar, "[",Mstar_low, ",", Mstar_high,"]";

# Source: http://turion.wordpress.com/2012/01/05/add-and-multiply-python-functions-operable-functions/
class normalized:
	def  __init__(self,f):
		self.f = f;
	def __call__(self,x):
		return self.f(x);
	def __mul__(self,other):
		return normalized(lambda x: other*self(x));
	
def LF(alpha,Mstar,Mmin,Mmax):
	#@normalized		
	def f(Mag):	
		return (10**(-0.4*(Mag-Mstar)*(alpha+1)))*(np.exp(-10**(-0.4*(Mag-Mstar))));
	norm = quad(f,Mmin,Mmax);	
	def g(Mag):
		return (10**(-0.4*(Mag-Mstar)*(alpha+1)))*(np.exp(-10**(-0.4*(Mag-Mstar))))/norm[0];
	return g;
	
lf = LF(alpha,Mstar,-25,Mmax);    
fig,ax = plt.subplots(1);
print type(lf); 
ax.plot(Mag,lf(Mag))
ax.hist(mi0[selection_index&(zphot0>=0.3)&(zphot0<0.4)&(mi0<Mcuts[0,3])&(mi0>-25)],50,histtype='step',normed=1,label=r'$z$ = 0.3 - 0.4');
ax.hist(mi0[selection_index&(zphot0>=z1a)&(zphot0<z1b)&(mi0<Mcuts[0,3])&(mi0>-25)&(mi0<Mmax)],50,weights=1.0/invW,histtype='step',normed=1,label=r'$z$ = 0.3 - 0.4 (weighted)');
ax.hist(mi0[selection_index&(zphot0>=0.65)&(zphot0<0.75)&(mi0<Mcuts[4,3])&(mi0>-25)],50,histtype='step',normed=1,label=r'$z$ = 0.65 - 0.75');
ax.hist(mi0[selection_index&(zphot0>=0.8)&(zphot0<0.85)&(mi0<Mcuts[6,3])&(mi0>-25)],50,histtype='step',normed=1,label = r'$z$ = 0.8 - 0.85');
ax.set_yscale('log');
ax.legend(loc=4).draggable();
plt.show();


