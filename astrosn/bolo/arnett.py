from dataclasses import asdict, dataclass, field
import json
from typing import Optional
import multiprocessing
import mpmath as mp
import numpy as np
from lmfit import minimize, Parameters, report_fit
from lmfit.minimizer import MinimizerResult
from numpy.typing import ArrayLike, NDArray

from astrosn.utils import PathType, StrEnum, FloatArr
from .bololc import BoloType, ModelFit, bolo_weighted_xyz


# Conversion Factors
days = 86400  # seconds
Msun = 1.98855e33  # grams
kms10 = 100000000.0  # 10000km/s -> cm/s

# Constants
Eni = 3.90e10  # ergs s-1 g-1
Eco = 6.78e9  # ergs s-1 g-1
tau_ni = 8.77  # *days #days * days =seconds
tau_co = 111.3  # *days #days * days =seconds

c = 2.998e10  # cm/s
Beta = 13.8  # constant
Kappa = 0.07  # cm2 g-1


# Derived quantities
# Diffusion Time
def get_tau_m(M_ej, v_ph):
    return (Kappa / (Beta * c)) ** (1 / 2) * (M_ej / v_ph) ** (1 / 2) / days  # taum in days


def get_s(tau_m):
    return tau_m * ((tau_co - tau_ni) / (2 * tau_co * tau_ni))  # days/days


def get_y(tau_m):
    return tau_m / (2 * tau_ni)  # unitless


def func_A(z, tau_m):
    y = get_y(tau_m)
    # return 2*z*np.exp((-2*z*y) + z**2)
    # print(2*z)
    TempA = mp.log(2 * z) - 2 * z * y + z ** 2
    # print(TempA)
    return mp.exp(TempA)


def func_B(z, tau_m):
    y = get_y(tau_m)
    s = get_s(tau_m)

    TempB = mp.log(2 * z) + 2 * z * s - 2 * z * y + z ** 2
    # print(TempB)
    return mp.exp(TempB)
    # return 2*z*np.exp(-2*z*y + 2*z*s + z**2)


def nonvector_L(t, *params):
    M_ej, M_ni, v_ph = params

    tau_m = get_tau_m(M_ej, v_ph)

    x = t / tau_m
    # Term1,er1=mp.quad(func_A,[0.0001,x],args=(tau_m))
    # Term2,er2=mp.quad(func_B,[0.0001,x],args=(tau_m))

    Term1 = mp.quad(lambda x: func_A(x, tau_m), [0.0001, x])
    Term2 = mp.quad(lambda x: func_B(x, tau_m), [0.0001, x])

    # lambda x: func_A(x, tau_m)

    # print M_ej/Msun,M_ni/Msun,v_ph/kms,Term1,Term2,get_y(tau_m),get_s(tau_m),tau_m,x

    lum = (M_ni * np.array(mp.exp(-x ** 2))) * ((Eni - Eco) * np.array(Term1) + (Eco * np.array(Term2))) / 1e43

    return lum * trapped(t * days, M_ej, v_ph)


L = np.vectorize(nonvector_L)


# EK_model*(trapped(xdata))
def get_Ek(M_ej, v_ph):
    return (3 / 10) * M_ej * (v_ph ** 2)


def tau(t, M_ej, Ek):
    T0 = np.sqrt((0.05305165 * Kappa * (M_ej ** 2)) / Ek)
    return (t / T0) ** (-2)


def trapped(t, M_ej, v_ph):
    Ek = get_Ek(M_ej, v_ph)
    return 1.0 - (0.965 * mp.exp(-tau(t, M_ej, Ek)))


def get_x_full(x: NDArray, E_exp: float) -> np.ndarray[float]:
    if E_exp == 0.0:
        x_full = x
    else:
        x_full = np.arange(x[0] - np.abs(E_exp) - 1.0, x[-1] + np.abs(E_exp), step=1.0)

    return x_full


def get_model_at_x(x, x_full, M, N, V, E_exp) -> np.ndarray[float]:
    model_now = L(x_full, M, N, V)

    model_now = model_now.astype(np.float64) * 1e43
    model_int = np.interp(x, x_full + E_exp, model_now)
    return model_int


def get_Ek_err(M_ej, M_ej_err, v_ph, v_ph_err):
    return get_Ek(M_ej, v_ph) * np.sqrt(((2*v_ph_err) / v_ph) ** 2 + (M_ej_err / M_ej) ** 2)


class RegularizedFit:

    def __init__(self, params: Parameters, varied_params: list[str],
                 lamb: float = 0):
        self.lamb = lamb
        self.params = params
        self.varied_params = varied_params
        self.regularization = 0

    def set_regularization(self, params, iter, resid, *args, **kws) -> None:
        n_samples = len(resid)
        param = np.array([params[p].value for p in self.varied_params])
        self.regularization = self.lamb * np.sum(param ** 2) / n_samples

    def regularized_least_squares_cost(self, r: ArrayLike) -> float:
        return np.sum(r ** 2) / 2 / len(r) + self.regularization


def process_params(params: Parameters) -> tuple[float, float, float, float]:
    mej = params['M_ej'].value * Msun
    mni = params['M_ni'].value * Msun
    e_exp = params['E_exp'].value
    vph = params['V_ph'].value * kms10
    return mej, mni, e_exp, vph


def get_model(params: Parameters, x: ArrayLike) -> NDArray[np.float64]:
    mej, mni, e_exp, vph = process_params(params)

    if abs(e_exp - 0.0) < 1e-3:
        # make +/- E_exp model range
        x_full = get_x_full(x, e_exp)
        x_full_p = x_full[x_full > 0.0]

        # calculate model within the full range shifted by E_exp
        model_int = get_model_at_x(x, x_full_p, mej, mni, vph, e_exp)
    else:
        model_now = L(x, mej, mni, vph)
        model_int = model_now.astype(np.float64) * 1e43

    return model_int

# fix velocity fit
def fit_func(params: Parameters, x: NDArray[np.float64], y: NDArray[np.float64],
             weights: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
    model_int = get_model(params, x)

    resids = model_int - y  # minimize this
    if weights is not None:
        resids = resids * weights
    return resids

@dataclass
class ArnettParams:
    mej: float
    vph: float
    e_exp: float = 0.0
    vary_e_exp: bool = True
    delta: float = 0.05

    e_exp_min: float = -3.0
    e_exp_max: float = 3.0
    e_exp_step: float = 0.2

    mej_min: float = 0.0
    mej_max: float = 10.0
    mej_step: float = 0.1

    delta_min: float = 0.00001
    delta_max: float = 0.99
    delta_step: float = 0.01

    @property
    def params(self) -> Parameters:
        params = Parameters()
        params.add('M_ej', value=self.mej, min=self.mej_min, max=self.mej_max,
                   vary=True, brute_step=self.mej_step)
        params.add('delta', value=self.delta, max=self.delta_max, min=self.delta_min, vary=True,
                   brute_step=self.delta_step)
        params.add('M_ni', expr='M_ej*delta', vary=False)
        params.add('E_exp', value=self.e_exp, max=self.e_exp_min, min=self.e_exp_max, vary=self.vary_e_exp,
                   brute_step=self.e_exp_step)
        params.add('V_ph', value=self.vph, vary=False)
        return params


class LMFitMethod(StrEnum):
    nelder = 'nelder'  # Nelder-Mead
    leastsq = 'leastsq'  # Levenberg-Marquardt
    least_squares = 'least_squares'  #  Trust Region Reflective
    bfgs = 'bfgs'  # Broyden-Fletcher-Goldfarb-Shanno
    brute = 'brute'  # brute force
    lbfgsb = 'lbfgsb'  # L-BFGS-B
    trust_constr = 'trust_constr'  # Trust Region Constrained
    slsqp = 'slsqp'  # Sequential Least SQuares Programming
    emcee = 'emcee'  # Markov Chain Monte Carlo


def fit_arnett(xdata: ArrayLike, ydata: ArrayLike, weights: Optional[ArrayLike] = None,
               arnett_params: ArnettParams = ArnettParams(1.0, 15, 0.0), regularization: float = 0,
               fit_emcee: bool = True, method: LMFitMethod | str = LMFitMethod.nelder) -> tuple[MinimizerResult, Optional[MinimizerResult]]:
    method = LMFitMethod(method)
    params = arnett_params.params
    fitter = RegularizedFit(params, ['M_ej', 'delta'], lamb=regularization)
    fit_result = minimize(fit_func, params, args=(xdata, ydata),
                          kws={'weights': weights}, method=method, nan_policy='omit',
                          iter_cb=fitter.set_regularization,
                          reduce_fcn=fitter.regularized_least_squares_cost)
    emcee_result = None
    if fit_emcee:
        # Explore posterior using emcee
        # fit_result.params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001), max=np.log(2))
        is_weighted = weights is not None
        emcee_result = minimize(fit_func, fit_result.params, args=(xdata, ydata),
                                kws={'weights': weights}, method='emcee',
                                nan_policy='omit', is_weighted=is_weighted, progress=False,
                                burn=300, steps=1000, thin=20,
                                workers=max(min(multiprocessing.cpu_count()//2 + 3, 10), 1))

    return fit_result, emcee_result


@dataclass
class  ArnettModelFit(ModelFit):
    name: str = "Arnett"
    params: Parameters = field(default_factory=Parameters)

    mej: float = field(default=-999)
    mej_err: float = field(default=-999)
    mni: float = field(default=-999)
    mni_err: float = field(default=-999)
    eexp: float = field(default=-999)
    eexp_err: float = field(default=-999)
    vph: float = field(default=-999)
    vph_err: float = field(default=-999)
    ek: float = field(default=-999)
    ek_err: float = field(default=-999)

    def __post_init__(self):
        if self.mej == -999:
            self.set_pars()
            self.set_errors()
        self.params = self.build_params()

    def with_error(self, parname: str):
        return f"${getattr(self, parname):.2f} \pm {getattr(self, f'{parname}_err'):.3f}$"

    def with_error_1(self, parname: str):
        return f"${getattr(self, parname):.1f} \pm {getattr(self, f'{parname}_err'):.2f}$"

    def __getitem__(self, __name: str) -> str:
        if __name in self.params:
            return f"${self.params[__name].value} \pm {self.params[__name].stderr}$"
        raise KeyError(f"Parameter {__name} not found")

    def set_pars(self) -> None:
        self.mej = self.params['M_ej'].value
        self.mni = self.params['M_ni'].value
        self.eexp = self.params['E_exp'].value
        self.vph = self.params['V_ph'].value

    def set_errors(self) -> None:
        self.mej_err = self.params["M_ej"].stderr
        self.mni_err = self.params["M_ni"].stderr
        self.eexp_err = self.params["E_exp"].stderr
        self.vph_err = self.params["V_ph"].stderr
        self.ek = get_Ek(self.mej, self.vph)
        self.ek_err = get_Ek_err(self.mej, self.mej_err, self.vph, self.vph_err)

    def build_params(self) -> Parameters:
        params = Parameters()
        params.add('M_ej', value=self.mej, vary=True)
        params.add('delta', value=self.mni/self.mej, vary=True)
        params.add('M_ni', expr='M_ej*delta', vary=False)
        params.add('E_exp', value=self.eexp, vary=False)
        params.add('V_ph', value=self.vph, vary=False)

        for par in params:
            if par == 'delta':
                continue
            params[par].stderr = getattr(self, f"{par.replace('_','').lower()}_err")

        return params

    def get_model(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return get_model(self.params, x)

    def to_json(self, path: PathType) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path: PathType) -> "ArnettModelFit":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def get_arnett_params(sn, phase_min: float = 0, phase_max: float = 10000, vph: float = 10., mej_init: float = 1) -> tuple[FloatArr, FloatArr, FloatArr, ArnettParams]:
    """Get the Arnett parameters for a SN
    vph: float = velocity of the ejecta in $10^4$ km/s
    mej_init: float = initial guess for the ejecta mass in solar masses
    """
    x, y, w = bolo_weighted_xyz(sn, BoloType.quasi)
    apars = ArnettParams(mej_init, vph=vph, vary_e_exp=False)

    mask = (x > phase_min) & (x < phase_max)
    return x[mask], y[mask], w[mask], apars
