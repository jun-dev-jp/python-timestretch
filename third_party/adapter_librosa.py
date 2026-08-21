import logging
from typing import override

from librosa.effects import time_stretch
from numpy import floating
from numpy.typing import NDArray

from third_party.base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class PV(BaseAdapter):
    def __init__(self, rate: float):
        """
        Initialize the PhaseVocoder with the given parameters.

        Args:
            rate: The rate of time stretching. (c.g. 1.0 = no change, 2.0 = double speed)

        Raises:
            ValueError: If the input parameters are invalid.
        """
        if rate < 0.0:
            raise ValueError("rate must be non-negative")

        self._rate: float = rate

    @override
    def __call__(self, x: NDArray[floating]) -> NDArray[floating]:
        """
        Apply Phase Vocoder algorithm to the input using pytsmod.

        Args:
            x (NDArray[floating]): The input audio signal.
                - shape must be 2D  e.g. (channels, samples)
                - dtype must be floating point
                - values must be in range [-1.0, 1.0]

        Returns:
            NDArray[floating]: The time-stretched audio signal.
            - dtype: same as input

        Raises:
            Exception: If an error occurs during the Phase Vocoder algorithm.
        """
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        try:
            return time_stretch(x, rate=self._rate)
        except Exception as e:
            logger.error(f"Error in librosa.effects.time_stretch: {e}")
            raise

    @property
    def rate(self) -> float:
        return self._rate
