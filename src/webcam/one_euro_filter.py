import math

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Initialize the One Euro Filter.
        
        Args:
            t0: Initial time
            x0: Initial value
            dx0: Initial derivative (velocity), default 0.0
            min_cutoff: Minimum cutoff frequency in Hz. Lower = more smoothing (less jitter) at low speed.
            beta: Speed coefficient. Higher = less lag (more responsiveness) at high speed.
            d_cutoff: Cutoff frequency for derivative smoothing (Hz).
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        Filter the signal.
        
        Args:
            t: Current timestamp
            x: Current value
            
        Returns:
            Filtered value
        """
        t_e = t - self.t_prev
        
        # Prevent division by zero or negative time
        if t_e <= 0.0:
            return self.x_prev
        
        # Filter the derivative (velocity)
        # The derivative is calculated from the raw signal change
        dx = (x - self.x_prev) / t_e
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)
        
        # The cutoff frequency is adaptive:
        # constant min_cutoff + coefficient * magnitude of velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter the signal
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
