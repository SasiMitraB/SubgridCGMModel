"""
ergane.units
~~~~~~~~~~~~~~~~~~
Physical unit system for Athena / AthenaK simulations.

Three base scales (length, density, velocity) fully determine all derived
quantities.  When attached to a ``SimulationData``, every field array returned
by field accessors and ``get_frame()`` is automatically multiplied by the
appropriate scale factor.

If no units are set, the default ``Units.code()`` applies scale factors of 1
everywhere and labels quantities as "[code units]".

Quick start
-----------
>>> from ergane.units import Units
>>>
>>> # Define a physical unit system for a GMC simulation
>>> cgs = Units(
...     length   = 3.086e18,    # 1 code length = 1 pc  (in cm)
...     density  = 1.67e-24,    # 1 code density = 1 proton mass / cc  (in g/cm³)
...     velocity = 1.0e5,       # 1 code velocity = 1 km/s  (in cm/s)
...     system   = "CGS",
...     labels   = {
...         "density":  "g cm⁻³",
...         "pressure": "dyn cm⁻²",
...         "velx":     "km s⁻¹",   # overrides derived default
...     },
... )
>>> sim.set_units(cgs)
>>> sim.density[0]          # array in g/cm³
>>> sim.units.label("velx") # "km s⁻¹"
"""

from __future__ import annotations

import math


# ── Unit label defaults (used when the user doesn't set explicit labels) ──────

_DERIVED_LABELS: dict[str, str] = {
    "density":  "ρ",
    "pressure": "P",
    "velx":     "v_x",
    "vely":     "v_y",
    "velz":     "v_z",
    "bx":       "B_x",
    "by":       "B_y",
    "bz":       "B_z",
    "time":     "t",
    "length":   "L",
}


# ── Units class ───────────────────────────────────────────────────────────────

class Units:
    """
    A self-consistent physical unit system defined by three base quantities.

    Parameters
    ----------
    length : float
        Number of physical length units per code length unit.
        E.g. ``3.086e18`` if 1 code length = 1 pc and the output is in cm.
    density : float
        Number of physical density units per code density unit.
    velocity : float
        Number of physical velocity units per code velocity unit.
    time : float, optional
        Override the derived time scale ``length / velocity``.
    pressure : float, optional
        Override the derived pressure scale ``density * velocity**2``.
    magnetic : float, optional
        Override the derived magnetic field scale ``sqrt(density) * velocity``
        (appropriate for Gaussian CGS where B is in Gauss).
    system : str
        Human-readable name, e.g. ``"CGS"``, ``"SI"``, ``"code"``.
    labels : dict[str, str], optional
        Map from field name → unit string for axis / colorbar labels.
        Any field not in this dict falls back to a generated label.

    Class methods
    -------------
    Units.code()
        Returns the trivial unit system with all scale factors = 1.
    Units.cgs(length, density, velocity)
        Convenience constructor with ``system="CGS"``.
    Units.si(length, density, velocity)
        Convenience constructor with ``system="SI"``.
    """

    def __init__(
        self,
        length:   float = 1.0,
        density:  float = 1.0,
        velocity: float = 1.0,
        time:     float | None = None,
        pressure: float | None = None,
        magnetic: float | None = None,
        system:   str = "custom",
        labels:   dict[str, str] | None = None,
        mu:       float = 0.62,
    ):
        self._length   = float(length)
        self._density  = float(density)
        self._velocity = float(velocity)
        self._time_override     = float(time)     if time     is not None else None
        self._pressure_override = float(pressure) if pressure is not None else None
        self._magnetic_override = float(magnetic) if magnetic is not None else None
        self.system  = system
        self.labels  = labels or {}
        self.mu      = float(mu)

    # ── Derived scale properties ──────────────────────────────────────────────

    @property
    def length(self) -> float:
        """Physical units per code length unit."""
        return self._length

    @property
    def density(self) -> float:
        """Physical units per code density unit."""
        return self._density

    @property
    def velocity(self) -> float:
        """Physical units per code velocity unit."""
        return self._velocity

    @property
    def time(self) -> float:
        """Physical units per code time unit (derived: length / velocity)."""
        if self._time_override is not None:
            return self._time_override
        return self._length / self._velocity

    @property
    def pressure(self) -> float:
        """Physical units per code pressure unit (derived: density × velocity²)."""
        if self._pressure_override is not None:
            return self._pressure_override
        return self._density * self._velocity ** 2

    @property
    def magnetic(self) -> float:
        """
        Physical units per code magnetic-field unit.

        Default derivation: sqrt(density × velocity²), appropriate for
        Gaussian-CGS where B is in Gauss and the 4π is absorbed into the
        code normalisation.  Override this if your simulation uses a
        different convention.
        """
        if self._magnetic_override is not None:
            return self._magnetic_override
        return math.sqrt(self._density * self._velocity ** 2)

    # ── Scale-factor lookup ───────────────────────────────────────────────────

    def scale(self, field: str) -> float:
        """
        Return the multiplicative scale factor for *field*.

        Multiplying a code-unit array by this factor converts it to physical
        units.  Returns 1.0 for unknown field names (safe no-op).
        """
        mapping: dict[str, float] = {
            "density":  self.density,
            "pressure": self.pressure,
            "velx":     self.velocity,
            "vely":     self.velocity,
            "velz":     self.velocity,
            "bx":       self.magnetic,
            "by":       self.magnetic,
            "bz":       self.magnetic,
        }
        return mapping.get(field, 1.0)

    # ── Label lookup ─────────────────────────────────────────────────────────

    def label(self, field: str) -> str:
        """
        Return a human-readable unit label for *field*.

        If ``labels[field]`` was set explicitly, that value is returned.
        Otherwise a generic label like ``"ρ [code]"`` is generated.
        """
        if field in self.labels:
            return self.labels[field]
        sym = _DERIVED_LABELS.get(field, field)
        if self.system == "code":
            return f"{sym} [code]"
        return f"{sym} [{self.system}]"

    # ── Class-method constructors ─────────────────────────────────────────────

    @classmethod
    def code(cls) -> "Units":
        """
        Trivial code-unit system: all scale factors = 1.
        
        This is the default when no units are set on a SimulationData.
        """
        return cls(
            length=1.0, density=1.0, velocity=1.0,
            system="code",
        )

    @classmethod
    def cgs(
        cls,
        length:   float,
        density:  float,
        velocity: float,
        pressure: float | None = None,
        **kwargs,
    ) -> "Units":
        """
        CGS convenience constructor. Equivalent to
        ``Units(length, density, velocity, system='CGS', ...)``.
        If velocity is passed in km/s (e.g. < 100) and pressure is omitted,
        pressure scale is automatically derived in CGS (dyn/cm²).
        """
        if pressure is None:
            v_cgs = velocity * 1e5 if velocity < 100.0 else velocity
            pressure = density * (v_cgs ** 2)
        return cls(
            length=length,
            density=density,
            velocity=velocity,
            pressure=pressure,
            system="CGS",
            **kwargs,
        )

    @classmethod
    def si(
        cls,
        length:   float,
        density:  float,
        velocity: float,
        **kwargs,
    ) -> "Units":
        """
        SI convenience constructor.  Equivalent to
        ``Units(length, density, velocity, system='SI', ...)``.
        """
        return cls(length=length, density=density, velocity=velocity,
                   system="SI", **kwargs)

    @classmethod
    def from_params(
        cls,
        params: dict[str, dict[str, str]],
        velocity_unit: str = "km/s",
    ) -> "Units":
        """
        Create a Units instance from the parsed simulation parameters (e.g. athinput).
        Looks for a 'units' section with length_cgs, mass_cgs, time_cgs, and mu.
        If no units section is present or if keys are missing, returns Units.code().
        """
        unit_section = params.get("units", {})
        if not unit_section:
            return cls.code()

        try:
            length_cgs = float(unit_section.get("length_cgs", 3.08568e18))
            mass_cgs = float(unit_section.get("mass_cgs", 4.91417e31))
            time_cgs = float(unit_section.get("time_cgs", 3.15576e13))
            mu = float(unit_section.get("mu", 0.62))

            density_scale = mass_cgs / length_cgs**3
            v_cgs = length_cgs / time_cgs

            if velocity_unit.lower() in ("km/s", "kms"):
                v_scale = v_cgs / 1e5
            else:
                v_scale = v_cgs

            return cls.cgs(
                length=length_cgs,
                density=density_scale,
                velocity=v_scale,
                pressure=density_scale * (v_cgs ** 2),
                mu=mu,
            )
        except (ValueError, TypeError, ZeroDivisionError):
            return cls.code()

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<Units  system='{self.system}'  "
            f"length={self._length:.3g}  "
            f"density={self._density:.3g}  "
            f"velocity={self._velocity:.3g}>"
        )
