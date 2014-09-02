import pymultinest
import pymultinest_schechterfit

import numpy as np
import scipy
from scipy import *
from scipy.special import *
import mpmath as mpm
import random as rnd

from pylab import *
import matplotlib.pyplot as plt

import os
from espeak import espeak

fig_estimates, ax_est = plt.subplots(2,1);
fig_estimates.suptitle('Estimated parameters vs Actual parameters');
ax_est_proxy = [ax_est[0],ax_est[1]];

ax_est[0].plot([-0.75,-0.25],[-0.75,-0.25],'g-.');
ax_est[1].plot([-25,-21], [-25,-21],'g-.');
#ax_est[1,0].plot([-2,2],[-2,2],'g-.');

ax_est[0].set_xlabel(r'$\alpha$',fontsize=20); ax_est[0].set_ylabel(r'$\hat{\alpha}$',fontsize=20,rotation=0,ha='right');
ax_est[1].set_xlabel(r'$M_*$',fontsize=20); ax_est[1].set_ylabel(r'$\hat{M_*}$',fontsize=20,rotation=0,ha='right');


#ax_est[1,0].set_xlabel(r'$q$',fontsize=20); ax[2].set_ylabel(r'$\hat{q}$',fontsize=20,rotation=0,ha='right');

Nmock=2;     n = 20000; # This value changes by a small amount later.

Alpha = np.linspace(-0.7,-0.3,Nmock); M0 = np.linspace(-24.0,-22.0,Nmock); np.random.shuffle(M0); Q = 0.0*np.linspace(0.6,1.1,Nmock); np.random.shuffle(Q);

#marginals for z
p = np.array([2.0,1.5,1.1,1.2,0.9,0.6,0.5,1.9,1.7,0.7,1.5,0.85,0.9,1.1]); p = p/p.sum();
zcenters = np.arange(0.325,1.025,0.05);

Z = [];
for l in xrange(len(zcenters)):
    Z.append([]);
    Z[l] = [(zcenters[l]+rnd.uniform(-0.025,0.025)) for r in xrange(int(round(n*p[l])))]; # no errors

Ndata = sum([int(round(n*p[l])) for l in xrange(len(zcenters))]); # correcting for round off errors in numbers
Ndata = [len(zj) for zj in Z];
n = np.array(Ndata).sum();

z = np.array([item for sublist in Z for item in sublist] ); #linearizes the non-rectangular list of lists

#Parameters of the conditional pdf
for i in xrange(Nmock):
    alpha = -0.7+ 0.4*np.random.uniform(); M0 =  -24.0 + 2.0*np.random.uniform(); q=-1.5+3.0*np.random.uniform(); 
    q = 0.0;
    #alpha = Alpha[i]; q = Q[i]; M0 = M0[i];
    params = np.array([alpha,M0, q]);
    Mmax = np.array(M0 + 5.0 - 2.0*zcenters); # 
    #Mmax = 0.0*Mmax; 
    # x = L/Lstar
    
    # !!!!!! NOTE TO SELF - CHECK THE NORMALIZATION, MIGHT NEED A 1/L* - done
    L = [[rnd.gammavariate(alpha+1,10**(-0.4*(M0+q*zi))) for zi in sublist] for sublist in Z];  # ???
    M = [[-2.5*np.log10(Lum) for Lum in sublist] for sublist in L];
    
    Mvl = [[m for m in M[l] if m<Mmax[-1]] for l in xrange(len(zcenters))]; #volume limited
    Lvl = [[l for l in L[j] if m<Mmax[-1]] for j in xrange(len(zcenters))];
    Zvl = [[z for z in Z[j] if m<Mmax[-1]] for j in xrange(len(zcenters))];
    Nvl = [len(mj) for mj in Mvl]; nvl = np.array(Nvl).sum();
    
    Mtest = [[m for m in M[l] if m<10.0] for l in xrange(len(zcenters))]; #volume limited
    Ltest = [[l for l in L[j] if m<10.0] for j in xrange(len(zcenters))];
    Ztest = [[z for z in Z[j] if m<10.0] for j in xrange(len(zcenters))];
    Ntest = [len(mj) for mj in Mtest]; ntest = np.array(Ntest).sum();

    Mbvl = [[m for m in M[l] if m<Mmax[l]] for l in xrange(len(zcenters))]; #bin-wise volume limited
    Lbvl = [[l for l in L[j] if m<Mmax[j]] for j in xrange(len(zcenters))];
    Zbvl = [[z for z in Z[j] if m<Mmax[j]] for j in xrange(len(zcenters))];
    Nbvl = [len(mj) for mj in Mbvl]; nbvl = np.array(Nbvl).sum();

    #print "Mbvl = ", Mbvl;
        
    sumL_bvl = [np.array(lj).sum() for lj in Lbvl ]; meanL_bvl = np.array(sumL_bvl).sum()/nbvl;
    sumM_bvl = [np.array(mj).sum() for mj in Mbvl ]; meanM_bvl = np.array(sumM_bvl).sum()/nbvl;
    sumZ_bvl = [np.array(zj).sum() for zj in Zbvl];
    
    sumM_vl = [np.array(mj).sum() for mj in Mvl]; meanM_vl = np.array(sumM_vl).sum()/nvl;
    sumL_vl = [np.array(lj).sum() for lj in Lvl]; meanL_vl = np.array(sumL_vl).sum()/nvl;
   
    sumM_test = [np.array(mj).sum() for mj in Mtest]; meanM_test = np.array(sumM_test).sum()/ntest;
    sumL_test = [np.array(lj).sum() for lj in Ltest]; meanL_test = np.array(sumL_test).sum()/ntest;

    #print "sumM as list: ", sumM;
    #print " L = ", L;
    #print " M = ", M;
 
    #while True: # generates mock data - samples of L (M) - no evolution
    #	    u = np.random.uniform(size=n);
    #	    a = gammaincinv(alpha+1,u);
    #	    M = Mstar-2.5*np.log10(a);
    #	    if ~(np.mean(10**(-0.4*M))==inf) & ~(np.mean(M)==inf) : # to avoid infinities
    #	    	break;
    
    #Mvl = np.delete(M,np.where(M>Mstar+2)); # volume limited
    #nvl = len(Mvl);
    
#    print "meanL = ", np.mean(10**(-0.4*M)), "meanM = ", np.mean(M);
    #raw_input("Press Enter to continue");
   
#    s = pymultinest_schechterfit.main(n,np.mean(10**(-0.4*M)),np.mean(M),Mmax);
    #svl =  pymultinest_schechterfit.main(nvl,np.mean(10**(-0.4*Mvl)),np.mean(Mvl),Mstar+2.0);

    #sbvl = pymultinest_schechterfit.evolve(np.array(nbvl),np.array(sumL),np.array(sumM),np.array(sumZ),np.array(Mmax));    
    #sbvl = pymultinest_schechterfit.main(n,meanL,meanM,Mmax[0]);

    fig, ax = plt.subplots(2,2);   
    fig.suptitle(r'ML Estimate of LF parameters for p'+str(i)+' with \n volume-limited sample size '+str(int(nvl))+' & bin-wise-volume-limited sample size '+str(int(nbvl))+'\n'+r' 1$\sigma$ errorbars' );  
   
    #svl = pymultinest_schechterfit.vl_2p(nvl,meanL_vl,meanM_vl,Mmax[-1],(fig,ax),true_params=(alpha,M0));

    s = pymultinest_schechterfit.vl_3p(np.array(Ntest),Ltest,np.array(sumM_test),Ztest,10.0);    
    svl = pymultinest_schechterfit.vl_3p(np.array(Nvl),Lvl,np.array(sumM_vl),Zvl,Mmax[-1],(fig,ax),true_params=(alpha,M0,q));
    #sbvl = pymultinest_schechterfit.bvl_3p(np.array(Nbvl),Lbvl,np.array(sumM_bvl),Zbvl,Mmax);
    #sbvl = pymultinest_schechterfit.bvl_2p(np.array(Nbvl),np.array(sumL_bvl),np.array(sumM_bvl),Mmax);
    
    print "True parameters = ", alpha, M0, q;
    print "Parameters = ", s['marginals'][0]['median'], s['marginals'][1]['median'], s['marginals'][2]['median']; 
    print "Parameters = ", svl['marginals'][0]['median'], svl['marginals'][1]['median'], svl['marginals'][2]['median']; 
    os.system('espeak "Yo bro"');
    
    [R,G,B] = np.random.uniform(size=3); # colors    
   
    for j in xrange(2):
        #m = svl['marginals'][j];
        #yj = m['median'];
        #low,high = m['1sigma'];
        #ax[j].errorbar(x=params[j],y=yj,yerr=np.transpose([[yj-low,high-yj]]),marker='s',color=(R,G,B),elinewidth=6,alpha=0.5);
        
        m = svl['marginals'][j];
        yj = m['median'];
        low, high = m['1sigma'];
        ax_est_proxy[j].annotate("p"+str(i),xy=(params[j],yj),xytext=(25,-5),textcoords='offset points',arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'));
        ax_est_proxy[j].errorbar(x=params[j], y=yj,yerr=np.transpose([[yj-low,high-yj]]),marker='o',color=(1-R,1-G,1-B),elinewidth=2,alpha=0.75);

    Mlin = np.array([m for mj in Mbvl for m in mj]);
    
    ax[1,1].hist(Mvl[0],50,histtype='step');
    ax[1,1].hist(Mvl[5],50,histtype='step');
    ax[1,1].hist(Mvl[-1],50,histtype='step');
    
    x = np.linspace(-30,10,401); a = 10**(-0.4*(Mlin-M0)); 

    #for j in xrange(2):
    #    m = svl['marginals'][j];
    #    yj = m['median'];
    #    low, high = m['3sigma'];
    #    ax[j].annotate("p"+str(i),xy=(params[j],yj),xytext=(25,-5),textcoords='offset points',arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'));
    #    ax[j].errorbar(x=params[j], y=yj,yerr=np.transpose([[yj-low,high-yj]]),marker='s',color=(R,G,B),label="n(p"+str(i)+"): "+str(nvl),elinewidth=2,alpha=0.6);

#ax[0,0].set_title(r'ML Estimate of $\alpha$ with \n volume-limited sample size '+str(int(nvl))+' & bin-wise-volume-limited sample size '+str(int(nbvl))+'\n'+r' 1$\sigma$ errorbars' );
#ax[0,1].set_title(r'ML Estimate of $M_*$ with \n volume-limited sample size '+str(int(nvl))+' & bin-wise-volume-limited sample size '+str(int(nbvl))+'\n'+r' 1$\sigma$ errorbars' );
#ax[1,0].set_title(r'ML Estimate of $q$ with \n volume-limited sample size '+str(int(nvl))+' & bin-wise-volume-limited sample size '+str(int(nbvl))+'\n'+r' 1$\sigma$ errorbars' );


#ax[0].legend(loc=2,ncol=2); #ax[1].legend(loc=2,ncol=2);
plt.show();
    	


