# Copyright (C) 2020 Jeremy Sanders <jeremy@jeremysanders.net>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import math
import numpy as N
from scipy.special import hyp2f1
import scipy.interpolate

from .physconstants import kpc_cm
from .par import Par, PriorGaussian, PriorBoundedGaussian
from . import utils

class Radii:
    """Define an equally-spaced set of shells/annuli and project between the two.
    :param rshell_kpc: width of shells/annuli in kpc
    :param num: number of shells/annuli

    """

    def __init__(self, rshell_kpc, num):
        self.num = num
        self.rshell_kpc = rshell_kpc

        self.edges_kpc = N.arange(num+1)*rshell_kpc
        self.inner_kpc = rin = self.edges_kpc[:-1]
        self.inner_cm = rin * kpc_cm
        self.outer_kpc = rout = self.edges_kpc[1:]
        self.outer_cm = rout * kpc_cm
        self.cent_kpc = 0.5*(rout+rin)
        self.cent_cm = self.cent_kpc * kpc_cm
        self.cent_logkpc = N.log(self.cent_kpc)
        # if shells are constant density, this is the mass-averaged radius
        self.massav_kpc = 0.75 * utils.diffQuart(rout, rin) / utils.diffCube(rout, rin)
        self.area_kpc2 = math.pi * utils.diffSqr(rout, rin)
        self.vol_kpc3 = (4/3*math.pi)*utils.diffCube(rout, rin)

        # matrix to convert from emissivity (per kpc3) to surface
        # brightness (per kpc2). projectionVolumeMatrix produces a
        # matrix which gives the total counts per annulus, so we want
        # to divide by the annulus area.
        self.proj_matrix = (
            utils.projectionVolumeMatrix(self.edges_kpc) / self.area_kpc2[:,N.newaxis] )
    def project(self, emissivity_pkpc3):
        """Project from emissivity profile to surface brightness (per kpc2)."""
        return self.proj_matrix.dot(emissivity_pkpc3).astype(N.float32)

class ProfileBase:
    def __init__(self, name, pars):
        self.name = name

    def compute(self, pars, radii):
        """Compute profile at centres of bins, given edges."""
        return N.zeros(radii.num)

    def prior(self, pars):
        """Return any prior associated with this profile."""
        return 0

class ProfileSum(ProfileBase):
    """Add one or more profiles to make a total profile."""

    def __init__(self, name, pars, subprofiles):
        ProfileBase.__init__(self, name, pars)
        self.subprofs = subprofiles

    def compute(self, pars, radii):
        compprofs = []
        for prof in self.subprofs:
            compprofs.append(prof.compute(pars, radii))
        total = N.sum(compprofs, axis=0)
        return total

    def prior(self, pars):
        return sum((prof.prior(pars) for prof in self.subprofs))

class ProfileFlat(ProfileBase):
    """Constant value profile."""

    def __init__(self, name, pars, defval=0., log=False, minval=-N.inf, maxval=N.inf):
        ProfileBase.__init__(self, name, pars)
        pars[name] = Par(defval, minval=minval, maxval=maxval)
        self.log = log

    def compute(self, pars, radii):
        v = pars[self.name].v
        if self.log:
            v = math.exp(v)
        return N.full(radii.num, v)

class ProfileBinned(ProfileBase):
    """Profile made up of constant values between particular radial edges.

    :param rbin_edges_kpc: array of bin edges, kpc
    :param defval: default value
    :param log: where to apply exp to output.
    """

    def __init__(self, name, pars, rbin_edges_kpc, defval=0., log=False):

        ProfileBase.__init__(self, name, pars)
        for i in len(rbin_edges_kpc)-1:
            pars['%s_%03i' % (name, i)] = Par(defval)
        self.rbin_edges_kpc = rbin_edges_kpc
        self.log = log

    def compute(self, radii):
        pvals = N.array([
            pars['%s_%03i' % (self.name, i)].v
            for i in range(radii.num)
            ])
        if self.log:
            pvals = N.exp(vals)
        idx = N.searchsorted(self.rbin_edges_kpc[1:], radii.cent_kpc)
        idx = N.clip(idx, 0, len(pvals)-1)

        # lookup bin for value
        outvals = pvals[idx]
        # mark values outside range as nan
        outvals[radii.outer_kpc < rbin_edges_kpc[0]] = N.nan
        outvals[radii.inner_kpc > rbin_edges_kpc[-1]] = N.nan

        return outvals

class ProfileInterpol(ProfileBase):
    """Create interpolated profile between fixed values

    :param rcent_kpc: where to interpolate between in kpc (sorted)
    :param defval: initial parameter values
    :param log: log profile
    :param extrapolate: extrapolate profile beyond endpoints (for linear)
    :param mode: interpolation type ('linear', 'cubic', 'quadratic'...)
    """

    def __init__(self, name, pars, rcent_kpc, defval=0., log=False, extrapolate=False, mode='linear'):

        ProfileBase.__init__(self, name, pars)
        for i in range(len(rcent_kpc)):
            pars['%s_%03i' % (name, i)] = Par(defval)
        self.rcent_logkpc = N.log(rcent_kpc)
        self.log = log
        self.extrapolate = extrapolate
        self.mode = mode

    def compute(self, pars, radii):
        pvals = N.array([
            pars['%s_%03i' % (self.name, i)].v
            for i in range(len(self.rcent_logkpc))
            ])
        if self.mode=='linear' and not self.extrapolate:
            vals = N.interp(radii.cent_logkpc, self.rcent_logkpc, pvals)
        else:
            f = scipy.interpolate.interp1d(
                self.rcent_logkpc, pvals, assume_sorted=True,
                kind=self.mode,
                fill_value='extrapolate')
            vals = f(radii.cent_logkpc)

        if self.log:
            vals = N.exp(vals)
        return vals

def _betaprof(rin_kpc, rout_kpc, n0, beta, rc_kpc):
    """Return beta function density profile

    Calculates average density in each shell.
    """

    # this is the average density in each shell
    # i.e.
    # Integrate[n0*(1 + (r/rc)^2)^(-3*beta/2)*4*Pi*r^2, r]
    # between r1 and r2
    def intfn(r_kpc):
        return (
            4/3 * n0 * math.pi * r_kpc**3 *
            hyp2f1(3/2, 3/2*beta, 5/2, -(r_kpc/rc_kpc)**2)
        )
    nav = (intfn(rout_kpc) - intfn(rin_kpc)) / (
        4/3*math.pi * utils.diffCube(rout_kpc,rin_kpc))
    return nav

class ProfileBeta(ProfileBase):
    """Beta model.

    Parameterised by:
    logn0: log_e of n0
    beta: beta value
    logrc: log_e of rc_kpc.
    """

    def __init__(self, name, pars):
        ProfileBase.__init__(self, name, pars)
        pars['%s_logn0' % name] = Par(math.log(1e-3), minval=-14., maxval=5.)
        pars['%s_beta' % name] = Par(2/3, minval=0., maxval=4.)
        pars['%s_logrc' % name] = Par(math.log(300), minval=-2, maxval=8.5)

    def compute(self, pars, radii):
        n0 = math.exp(pars['%s_logn0' % self.name].v)
        beta = pars['%s_beta' % self.name].v
        rc_kpc = math.exp(pars['%s_logrc' % self.name].v)

        prof = _betaprof(
            radii.inner_kpc, radii.outer_kpc,
            n0, beta, rc_kpc)
        return prof

class ProfileVikhDensity(ProfileBase):
    """Density model from Vikhlinin+06, Eqn 3.

    Modes:
    'double': all components
    'single': only first component
    'betacore': only first two terms of 1st cmpt (beta, with powerlaw core)

    Densities and radii are are log base 10
    """

    def __init__(self, name, pars, mode='double', freers=True, freer2=True):
        ProfileBase.__init__(self, name, pars)
        self.mode = mode

        pars['%s_logn0_1' % name] = Par(-3.00, minval=-8.00, maxval=5.00, soft=True)
        pars['%s_beta_1'  % name] = Par( 0.60, minval= 0.00, maxval=4.00, soft=True)
        pars['%s_logrc_1' % name] = Par( 2.50, minval= 0.00, maxval=5.00, soft=True)
        pars['%s_alpha'   % name] = Par( 0.00, minval=-2.00, maxval=4.00, soft=True)
        
        if mode in {'single', 'double'}:
            pars['%s_epsilon' % name] = Par(3.00, minval=0.00, maxval=5.00)
            pars['%s_gamma'   % name] = Par(3.00, minval=0.00, maxval=5.00, frozen=True)
            if freers: 
                pars['%s_logr_s'  % name] = Par(2.50, minval=0.00, maxval= 5.00)
            else:
                pars['%s_logc_s'  % name] = Par(0.00, minval=0.00, maxval= 2.00)
            
        if mode == 'double':
            pars['%s_logn0_2' % name] = Par(-2.50, minval=-8.00, maxval=5.00, soft=True)
            pars['%s_beta_2'  % name] = Par( 0.50, minval= 0.00, maxval=4.00, soft=True)
            if freer2:
                pars['%s_logrc_2' % name] = Par( 3.90, minval=-2.00, maxval=5.00, soft=True)
            else:
                pars['%s_logc_2' % name] = Par(-2.00, minval=-6.00, maxval=0.00, soft=True)

    def compute(self, pars, radii):
        n0_1   = 10**pars['%s_logn0_1' % self.name].v
        rc_1   = 10**pars['%s_logrc_1' % self.name].v
        beta_1 = pars['%s_beta_1' % self.name].v
        alpha  = pars['%s_alpha'  % self.name].v

        r = radii.cent_kpc
        retn_sqd = (
            n0_1**2 *
            (r/rc_1)**(-alpha) / (
                (1+r**2/rc_1**2)**(3*beta_1-0.5*alpha))
            )

        if self.mode in ('single', 'double'):
            if '%s_logr_s'%self.name in pars.keys():
                r_s = 10**pars['%s_logr_s' % self.name].v
            elif '%s_logc_s'%self.name in pars.keys():
                r_s = rc_1*(10**pars['%s_logc_s' % self.name].v)

            epsilon = pars['%s_epsilon' % self.name].v
            gamma   = pars['%s_gamma'   % self.name].v

            retn_sqd /= (1+(r/r_s)**gamma)**(epsilon/gamma)

        if self.mode == 'double':
            n0_2 = 10**pars['%s_logn0_2' % self.name].v
            if '%s_logrc_2'%self.name in pars.keys():
                rc_2 = 10**pars['%s_logrc_2' % self.name].v
            elif '%s_logc_2'%self.name in pars.keys():
                rc_2 = rc_1*(10**pars['%s_logc_2' % self.name].v)
            beta_2 = pars['%s_beta_2' % self.name].v

            retn_sqd += n0_2**2 / (1 + r**2/rc_2**2)**(3*beta_2)

        ne = N.sqrt(retn_sqd)
        return ne

class ProfileMcDonaldT(ProfileBase):
    """Temperature model from McDonald+14, equation 1

    Log values are are log_e
    """

    def __init__(self, name, pars):
        ProfileBase.__init__(self, name, pars)

        pars['%s_logT0' % name] = Par(0.50, minval=-2.00, maxval=1.70)
        pars['%s_cmin'  % name] = Par(0.50, minval= 0.00, maxval=1.50)
        pars['%s_logrc' % name] = Par(2.50, minval=-1.00, maxval=5.00)
        pars['%s_logrt' % name] = Par(2.75, minval=-1.00, maxval=8.00)
        pars['%s_acool' % name] = Par(2.00, minval= 0.00, maxval=8.50)
        pars['%s_a'     % name] = Par(0.00, minval=-4.00, maxval=4.00)
        pars['%s_b'     % name] = Par(1.00, minval= 0.01, maxval=4.00)
        pars['%s_c'     % name] = Par(1.00, minval= 0.00, maxval=4.00)

    def compute(self, pars, radii):
        n = self.name
        T0    = 10**pars['%s_logT0' % n].v
        rc    = 10**pars['%s_logrc' % n].v
        rt    = 10**pars['%s_logrt' % n].v
        acool = pars['%s_acool'     % n].v
        cmin  = pars['%s_cmin'      % n].v
        a     = pars['%s_a'         % n].v
        b     = pars['%s_b'         % n].v
        c     = pars['%s_c'         % n].v

        x = radii.cent_kpc
        x_rc = x*(1/rc)
        x_rt = x*(1/rt)

        T = (
            T0
            * (x_rc**acool + cmin)
            / (1 + x_rc**acool)
            * x_rt**(-a)
            / (1 + x_rt**b)**(c/b)
            )

        return T
    
class ProfileVikhT500(ProfileBase):
    """Temperature model from Vikhlinin+06, Eqn 4., with r in r500 units"""

    def __init__(self, name, pars):
        ProfileBase.__init__(self, name, pars)

        pars['%s_logT0' % name] = Par(0.500, minval=-2.00, maxval=1.70)
        pars['%s_cmin'  % name] = Par(0.500, minval= 0.00, maxval=1.50)
        pars['%s_xc'    % name] = Par(0.045, minval= 0.00, maxval=2.00)
        pars['%s_xt'    % name] = Par(0.600, minval= 0.00, maxval=2.00)
        pars['%s_acool' % name] = Par(1.900, minval= 0.00, maxval=8.50)
        pars['%s_a'     % name] = Par(0.000, minval=-4.00, maxval=4.00)
        pars['%s_b'     % name] = Par(2.000, minval= 0.01, maxval=4.00)
        pars['%s_c'     % name] = Par(0.900, minval= 0.00, maxval=4.00)

        self.use500 = True

    def compute(self, pars, radii, r500=1.00):
        n = self.name
        T0    = 10**pars['%s_logT0' % n].v
        xc    = pars['%s_xc'        % n].v
        xt    = pars['%s_xt'        % n].v
        acool = pars['%s_acool'     % n].v
        cmin  = pars['%s_cmin'      % n].v
        a     = pars['%s_a'         % n].v
        b     = pars['%s_b'         % n].v
        c     = pars['%s_c'         % n].v

        x = radii.cent_kpc
        x_rc = (x/r500/xc)**acool
        x_rt =  x/r500/xt

        return T0 * ((x_rc + cmin)/ (1 + x_rc)) * x_rt**(-a) / (1 + x_rt**b)**(c/b)
            

class ProfileK(ProfileBase):
    def __init__(self, name, pars):
        ProfileBase.__init__(self, name, pars)
        pars['%s_logK0' % name] = Par(0.50, minval=-4.00, maxval=3.50)
        pars['%s_logKc' % name] = Par(0.50, minval=-2.00, maxval=4.00)
        pars['%s_logrc' % name] = Par(2.00, minval= 0.00, maxval=5.00)
        pars['%s_ac'    % name] = Par(1.10, minval= 0.00, maxval=5.00)

        self.useK = True

    def compute(self, pars, radii, nr):
        n = self.name
        K0 = 10**pars['%s_logK0' % n].v
        Kc = 10**pars['%s_logKc' % n].v
        rc = 10**pars['%s_logrc' % n].v
        ac = pars['%s_ac'        % n].v 

        Kr = K0+Kc*(radii.cent_kpc/rc)**ac
        return Kr*(nr**(2.00/3.00))