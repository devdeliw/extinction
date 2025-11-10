import emcee 
import corner 
import numpy as np 
import matplotlib.pyplot as plt 

plt.rcParams["font.family"] = "serif" 
plt.rcParams["mathtext.fontset"] = "cm" 

def _sigmoid(x): 
    return np.where( 
        x >= 0, 
        1.0 / (1.0 + np.exp(-x)), 
        np.exp(x) / (1.0 + np.exp(x))
    )

class MCMC: 
    """
    emcee sampler for compound gaussian+linear histogram model 

    theta = (u, amplitude, mu, sigma, m, b)
    f_RC = sigmoid(u) in (0, 1) 
    """

    def __init__(
        self, 
        data: np.ndarray, 
        bins     : int  = 50, 
        nwalkers : int  = 64, 
        nsteps   : int  = 15000, 
        burnin   : int  = 1000, 
        thin     : int  = 1,
        progress : bool = True
    ):
        self.data = np.asarray(data) 
        self.bins = bins 
        
        self.nwalkers = nwalkers 
        self.nsteps   = nsteps 
        self.burnin   = burnin 
        self.thin     = thin 

        self.bin_heights, self.bin_edges = np.histogram(self.data, bins=bins)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_errors  = np.sqrt(self.bin_heights + 1) 

        self.samples     = None 
        self.best_params = None 
        self.log_probs   = None

        self.progress = progress 

    def _ansatz(self): 
        u0   = np.log(0.1 / 0.9) 
        amp0 = self.bin_heights.max() / 4 
        mu0  = np.mean(self.data) 
        sig0 = np.std(self.data) 
        m0   = 0.0 
        b0   = self.bin_heights.min() if self.bin_heights.min() > 0 else 1.0 

        return [u0, amp0, mu0, sig0, m0, b0] 

    @staticmethod 
    def _compound_model(theta, x): 
        u, amp, mu, sig, m, b = theta 
        f_rc = _sigmoid(u) 
        f_rc = np.clip(f_rc, 1e-6, 1 - 1e-6)

        # compound gaussian + linear 
        gaussian = amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
        linear   = m * x + b 
        
        return f_rc * gaussian + (1.0 - f_rc) * linear 

    def _log_prior(self, theta): 
        u, amp, mu, sig, m, b = theta 
        mu0, s0 = np.mean(self.data), np.std(self.data) 

        alpha, beta = 3.0, 2.0 
        f_rc = _sigmoid(u) 
        f_rc = np.clip(f_rc, 1e-6, 1 - 1e-6) 

        ln_prior  = (alpha - 1) * np.log(f_rc) + (beta - 1) * np.log(1 - f_rc)
        ln_prior += np.log(f_rc) + np.log(1 - f_rc) # jacobian

        if not np.isfinite(ln_prior): 
            return -np.inf 
        if not (0 < amp < 1000): 
            return -np.inf 
        if not (mu0 - 2.0 < mu < mu0 + 2.0): 
            return -np.inf 
        if not (max(0.0, s0 - 0.5) < sig < s0 + 0.5): 
            return -np.inf 
        if m > 30 or b > 50: 
            return -np.inf 

        return ln_prior 

    def _log_likelihood(self, theta): 
        model = self._compound_model(theta, self.bin_centers) 
        resid = self.bin_heights - model 
        ivar  = 1.0 / self.bin_errors ** 2 
        return -0.5 * np.sum(resid**2 * ivar + np.log(2*np.pi / ivar)) 

    def _log_probability(self, theta): 
        lp = self._log_prior(theta) 
        return -np.inf if not np.isfinite(lp) else lp + self._log_likelihood(theta) 

    def run(self): 
        nwalkers = self.nwalkers 
        nsteps   = self.nsteps 
        burnin   = self.burnin
        thin     = self.thin 

        p0   = self._ansatz() 
        ndim = len(p0)

        # initialize walker pos 
        pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim) 

        # run 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_probability) 
        sampler.run_mcmc(pos, nsteps, progress=self.progress) 

        self.chain     = sampler.get_chain(discard=burnin, flat=False) 
        self.log_probs = sampler.get_log_prob(discard=0, flat=False) 
        self.samples   = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        # medians 
        u_m, amp_m, mu_m, sig_m, m_m, b_m = np.median(self.samples, axis=0) # type: ignore
        f_m = _sigmoid(u_m)

        self.best_params = { 
            "frac_RC"   : f_m, 
            "amplitude" : amp_m, 
            "mean"      : mu_m, 
            "stddev"    : sig_m, 
            "slope"     : m_m, 
            "intercept" : b_m, 
        }

        return self.best_params, self.samples, self.log_probs

    def autocorrelation(self, n=200, c=5, tol=50, quiet=True): 
        # track tau as chain length N grows 
        # store curve for plotting 

        if self.chain is None: 
            raise RuntimeError("run the MCMC first") 

        n_steps = self.chain.shape[0] 
        N_vals  = np.linspace(10, n_steps, n, dtype=int) 
        means   = [] 

        for N in N_vals: 
            tau = emcee.autocorr.integrated_time( 
                self.chain[:N], c=c, tol=tol, quiet=quiet
            )
            means.append(np.mean(tau)) 

        self.acorr = (N_vals, np.asarray(means)) 
        return self.acorr

    def plot_corner(self, bins=50, smooth=1.0):
        if self.samples is None:
            raise RuntimeError("run the MCMC first")
            
        labels = [r"$u$", r"$A$", r"$\mu$", r"$\sigma$", r"$m$", r"$b$"]
        fig = corner.corner(
            self.samples,
            labels=labels,
            bins=bins,
            smooth=smooth,
            show_titles=True,
            quantiles=[0.16, 0.50, 0.84],
            title_kwargs={"fontsize": 14}, 
            label_kwargs={"fontsize": 18}, 
        )
        return fig

    def plot_fit(self): 
        if self.best_params is None: 
            raise RuntimeError("run the MCMC first") 

        fig, ax = plt.subplots(1, 1, figsize=(8, 6)) 
        plt.errorbar( 
            self.bin_centers, self.bin_heights, 
            yerr=self.bin_errors, fmt=".k", capsize=2 
        )

        bf = self.best_params 
        u  = np.log(bf["frac_RC"] / (1.0 - bf["frac_RC"]))
        theta = [ 
            u, 
            bf["amplitude"], 
            bf["mean"], 
            bf["stddev"], 
            bf["slope"], 
            bf["intercept"] 
        ]

        x = np.linspace(self.bin_edges[0], self.bin_edges[-1], 500) 
        plt.plot(x, self._compound_model(theta, x), "r-") 
        plt.xlabel("bin center", fontsize=15)
        plt.ylabel("count", fontsize=15) 

        return fig, ax

        
