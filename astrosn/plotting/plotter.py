import inspect
from typing import Any, Optional, cast

from dataclasses import replace
from lmfit.minimizer import MinimizerResult

from astrosn.photometry import Photometry
from astrosn.supernova import SN

from .matplotlib_classes import Axes, Figure, Legend
from .parameters import AxisParameters, PlotParameters
from .settings import PlotSettings
from .defaults import DefaultSettings, MaxSettings
from .plot_lc import (
    PhotPlotTypes,
    PlotType,
    SNPlotTypes,
    format_axis,
    get_site_markers,
    get_unsub_mark,
    label_axis,
    make_plot_colors,
    make_plot_shifts,
    make_single_figure,
    plot_emcee,
    update_legend_handles_labels,
)


class Plotter:
    """Class based plotting."""

    def __init__(
        self,
        sn: SN,
        plot_type: PlotType = SNPlotTypes.ABS_MAG,
        params: PlotParameters = PlotParameters(),
        axis_params: AxisParameters = AxisParameters(),
        defaults: PlotSettings = DefaultSettings(),
        **plot_kwargs: Any,
    ):
        self.sn = sn
        self.sn.sites.markers = cast(dict[int, str | None], get_site_markers(sn.sites.markers))
        self.plot_type = plot_type
        self.signature = list(inspect.signature(self.plot_type).parameters.keys())
        self.plot_params = params
        self.plot_kwargs = plot_kwargs
        self.axis_params = axis_params

    def update_plot_type(self, plot_type: PlotType, **plot_kwargs: Any):
        self.plot_type = plot_type
        self.signature = list(inspect.signature(plot_type).parameters.keys())
        self.plot_kwargs = plot_kwargs

    def get_sn_or_phot(self) -> SN | Photometry:
        if self.plot_type in SNPlotTypes.values():
            return self.sn
        elif self.plot_type in PhotPlotTypes.values():
            lims = self.plot_type == PhotPlotTypes.LIM
            return self.sn.band(
                self.plot_params.band.name,
                site=self.plot_params.site,
                return_absmag=self.plot_params.absmag,
                flux=self.plot_params.as_flux,
                lims=lims,
            )
        raise ValueError(f"Plot type {self.plot_type} not recognised")

    def get_plot_params(self) -> dict[str, Any]:
        kwargs = {k: v for k, v in self.plot_params.items() if k in self.signature}
        return kwargs | self.plot_kwargs

    def save_current_legend(self, ax: Axes):
        if ax.get_legend() is not None:
            h, l = ax.get_legend_handles_labels()
            if self.axis_params.custom_labels is None:
                self.axis_params.custom_labels = l
            if self.axis_params.custom_handles is None:
                self.axis_params.custom_handles = h
            ax, h, l = update_legend_handles_labels(ax, self.axis_params.custom_handles, self.axis_params.custom_labels)
            self.axis_params.custom_handles = h
            self.axis_params.custom_labels = l

    def plot(self, ax: Optional[Axes] = None, **fig_kwargs: Any) -> tuple[Figure | None, Axes]:
        kwargs = self.get_plot_params()
        arg = self.get_sn_or_phot()
        fig = None
        if ax is None:
            fig, ax = make_single_figure(**fig_kwargs)
        ax = self.plot_type(arg, ax=ax, **kwargs)
        self.save_current_legend(ax)
        return fig, ax

    def label_axis(
        self,
        ax: Axes,
        xlabel: str,
        ylabel: str,
        legend: bool = True,
        **legend_kwargs: Any,
    ) -> tuple[Axes, Optional[Legend]]:
        self.axis_params = replace(
            self.axis_params,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            legend_kwargs=legend_kwargs,
        )
        return label_axis(ax, xlabel, ylabel, legend, **legend_kwargs)

    def format_axis(
        self,
        ax: Axes,
        invert: bool = False,
        log: bool = False,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        x_tick_interval: Optional[float] = None,
        y_tick_interval: Optional[float] = None,
        x_minors: Optional[int] = 5,
        y_minors: Optional[int] = 5,
    ) -> Axes:
        self.axis_params = replace(
            self.axis_params,
            invert=invert,
            log=log,
            xlim=xlim,
            ylim=ylim,
            x_tick_interval=x_tick_interval,
            y_tick_interval=y_tick_interval,
            x_minors=x_minors,
            y_minors=y_minors,
        )
        return format_axis(
            ax,
            invert,
            log,
            xlim,
            ylim,
            x_tick_interval,
            y_tick_interval,
            x_minors,
            y_minors,
        )

    def plot_emcee(self, res: MinimizerResult):
        return plot_emcee(res)

    def setup_plot(self):
        sn = self.sn
        # Can also set automatic plot shifts & colors, but here using custom ones.
        sn = make_plot_shifts(
            sn=sn,
            reversed_for_abs=self.plot_params.absmag,
            shifts=self.plot_params.shifts,
        )
        sn = make_plot_colors(sn=sn, colors=self.plot_params.colors)
        self.sn = sn

    def legend_additions(self, ax: Axes) -> None:
        if self.plot_params.mark_unsub:
            handles, labels = get_unsub_mark()
            ax, handles, labels = update_legend_handles_labels(ax, handles, labels)
            if self.axis_params.custom_labels is None:
                self.axis_params.custom_handles = handles
                self.axis_params.custom_labels = labels
            else:
                for handle, label in zip(handles, labels):
                    if label not in self.axis_params.custom_labels:
                        self.axis_params.custom_labels.append(label)
                        if self.axis_params.custom_handles is None:
                            raise ValueError(
                                "Custom handles should not be None when custom labels are"
                                f" {self.axis_params.custom_labels}"
                            )
                        self.axis_params.custom_handles.append(handle)

    def __call__(self, ax: Optional[Axes] = None, **fig_kwargs: Any) -> tuple[Optional[Figure], Axes]:
        self.setup_plot()
        fig, ax = self.plot(ax, **fig_kwargs)

        ax = self.axis_params.format_axis(ax)
        return fig, ax
