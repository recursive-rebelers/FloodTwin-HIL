from abc import ABC, abstractmethod

class BaseSensor(ABC):
    def __init__(self, name, config):
        self.name = name
        self.config = config

    @abstractmethod
    def generate(self, carla_sensor_data):
        """Standardize incoming raw CARLA data."""
        pass

    @abstractmethod
    def add_noise(self, data):
        """Inject environmental noise (e.g., thermal, turbidity)."""
        pass

    @abstractmethod
    def inject_fault(self, data, fault_type):
        """Simulate hardware failure or signal blockage."""
        pass

    @abstractmethod
    def get_reliability(self, data):
        """Calculate confidence score (0.0 to 1.0)."""
        pass