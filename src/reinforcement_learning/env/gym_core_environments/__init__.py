try:
    import Box2D
    from .car_racing import CarRacing
except ImportError:
    Box2D = None
