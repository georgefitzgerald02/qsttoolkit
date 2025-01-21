import cmath
import random


def _range_error(magnitude_range: list[float, float], integers=False, positive=True) -> None:
    """Basic error checking for the range of magnitudes."""
    if len(magnitude_range) != 2:
        raise ValueError("magnitude_range must be a list of two floats")
    if magnitude_range[0] > magnitude_range[1]:
        raise ValueError("min magnitude must be less than max magnitude")
    if positive:
        if magnitude_range[0] < 0:
            raise ValueError("min magnitude must be greater than or equal to 0")
    if integers:
        if not float(magnitude_range[0]).is_integer() or not float(magnitude_range[1]).is_integer():
            raise ValueError("min and max magnitudes must be integers")

def _random_complex(magnitude_range: list[float, float]) -> complex:
    """Generates a random complex number with a given magnitude range."""
    # Error checking
    _range_error(magnitude_range)
    
    magnitude = random.uniform(magnitude_range[0], magnitude_range[1])      # Is uniform best?? Options?
    angle = random.uniform(0, 2 * cmath.pi)
    return cmath.rect(magnitude, angle)

def _get_var_name(var: any) -> str:
    """Returns the name of a variable."""
    for name, value in globals().items():
        if value is var:
            return name