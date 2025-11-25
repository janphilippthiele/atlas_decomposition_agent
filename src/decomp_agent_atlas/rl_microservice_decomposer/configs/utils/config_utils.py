from dataclasses import dataclass, asdict
from abc import ABC

@dataclass
class BaseConfig(ABC):
    """Base class for all configurations"""
    
    def to_dict(self):
        """Convert config object to dictionary for JSON serialization"""
        return asdict(self)