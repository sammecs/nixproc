from functools import wraps
import numpy as np
import uncertainties as uc


# Idea:
# write a function: dict a -> dict b
# where a['classname'] is a constructor which takes a and returns an element
# and b['classname'] is the element returned by calling a['classname'] with a
# Only use with python 3.6 or later, because otherwise order of dicts is not
# guaranteed, which is essential to build the collection!


def realize(d, infer_keys=False, start_with=dict()):
    if infer_keys:
        d = dict(map(lambda f: (f.__name__.lower(), f), d))
    # d - input dict
    # r - output dict
    r = start_with
    for key, entry in d.items():
        # entry is either a dict or a function or a class
        if callable(entry):
            # call function or class constructor
            r[key] = entry(r)
        elif isinstance(entry, dict):
            r[key] = realize(entry)
        else:
            raise TypeError("Expected a callable or dictionary, but got {}"
                            .format(type(entry)))
    return r


# see https://stackoverflow.com/questions/3012421/python-memoising-deferred-lookup-property-decorator
def lazyprop(func):
    attr_name = '_' + func.__name__

    @property
    @wraps(func)  # conserves docstrings
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazyprop


def labeled_curve_fit(fit_fn, guess_params, xdata, ydata=None, xwindow=None,
                      sigma=None, **kwargs):
    """Make a curve_fit for the given data, but cool.

    Parameters:
      fit_fn: function of the form fit_fn(x, p1, p2, ...), for which the
              parameters (p1, p2, ...) should be found.
      guess_params: Initial guesses for the parameters (p1, p2, ...) in a
                    dictionary format, i.e. {'p1': g1, 'p2': g2}
      xdata, ydata: Data points. If ydata == None, assume a pd.Series was given.
                    Then, assume you meant (xdata.index, xdata) as data points.
      xwindow: Select only data points with min(xwindow) <= x <= max(xwindow)
               for the fit.
      **kwargs: Passed to curve_fit.
    Returns:
      A dictionary with resulting values, i.e.
      {'p1': ufloat(1, 0.2), 'p2': ufloat(2, 0.3)}"""
    # Assume a pd.Series was passed if ydata is None
    if ydata is None:
        xdata, ydata = xdata.index, xdata
    if xwindow is None:
        xwindow = min(xdata), max(xdata)
    argnames = inspect.getargspec(fit_fn).args[1:]
    p0 = list(map(lambda n: guess_params.get(n, 1), argnames))
    mask = np.logical_and(min(xwindow) <= xdata, xdata <= max(xwindow))
    xdata, ydata = xdata[mask], ydata[mask]
    if sigma is not None:
        sigma = sigma[mask]
    popt, pcov = curve_fit(fit_fn, xdata, ydata, p0=p0, sigma=sigma, **kwargs)
    popt_uc = uc.correlated_values(popt, pcov)
    return dict(zip(argnames, popt_uc))


def plot_spectrum(ax, x, y, **kwargs):
    """Ruft ax.plot auf. Die übergebene Linie besteht allerdings nicht direkt
aus den (x,y)-Daten. Stattdessen wird eine Linie generiert, die ein Histogramm
(mit den Säulenmitten auf x, Säulenhöhen y) umreißt. Weitere keyword-arguments
werden direkt an ax.plot weitergegeben."""
    # Alternativ zu ax.plot wäre auch ax.hist denkbar, dann müsste man sich
    # aber darum kümmern, dass das Histogramm nicht gefüllt wird und nur die
    # Outline sichtbar ist

    # Potentiell wichtig: Bei nicht-linear verteilten x-Werten liegen die
    # x-Werte nicht in der Mitte der Säulen. Das kann zu irreführenden
    # Diagrammen führen. Da wir diesen Anwendungsfall gerade nicht betrachten,
    # habe ich diese Funktionalität nicht implementiert.

    # bins sind die Positionen der Kanten der einzelnen Bars
    # also meistens einfach in der Mitte zwischen zwei Datenpunkten
    inner_bin_edges = (x[1:] + x[:-1]) / 2
    bins = np.empty(len(x) + 1)
    # Position der äußeren Kanten wird so gesetzt, dass der x[0] in der Mitte
    # zwischen bins[0] und bins[1] ist
    bins[0] = (3*x[0] - x[1]) / 2
    bins[-1] = (3*x[-1] - x[-2]) / 2
    bins[1:-1] = inner_bin_edges
    # für jede Kante bekommt die Linie zwei Punkte
    hist_x = np.repeat(bins, 2) # length: 2*(len(x)+1)
    hist_y = np.empty(2*(len(x) + 1))
    # Am Anfang und am Ende soll die Histogrammlinie auf 0 gehen
    hist_y[0] = 0
    hist_y[-1] = 0
    hist_y[1:-1] = np.repeat(y, 2)
    ax.plot(hist_x, hist_y, **kwargs)


def data_margin(window, fraction):
    margin = (max(window) - min(window)) * fraction
    return min(window) - margin, max(window) + margin


def plot_fit(ax, window, func, params, num=1000, **kwargs):
    """Plot a fit function.

    Parameters: ax: Axis to plot on
                window = (xmin, xmax): in which range the fit should be
                                       displayed
                func: Function that describes the fit
                params: Parameters supplied to the function
                        i.e. func(x, *params) is printed
                kwargs: supplied to ax.plot"""
    fitx = np.linspace(*window, num=num)
    fity = func(fitx, *nominal_values(params))
    ax.plot(fitx, fity, **kwargs)


def corresponding_twinax(ax, conversion_func, is_linear=False, which='x'):
