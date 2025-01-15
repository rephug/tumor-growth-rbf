from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np

@dataclass
class TreatmentEvent:
    """Represents a single treatment event."""
    time: float  # Time in days
    treatment_type: str
    dose: float
    duration: Optional[float] = None
    constraints: Optional[Dict] = None

class TreatmentScheduler:
    """
    Handles complex treatment scheduling including:
    - Multiple concurrent treatments
    - Treatment constraints and dependencies
    - Dose timing optimization
    """
    
    def __init__(self, treatment_module):
        self.treatment_module = treatment_module
        self.schedule: List[TreatmentEvent] = []
        self.current_time = 0.0
        
    def add_radiotherapy_course(self,
                              start_time: float,
                              total_dose: float,
                              num_fractions: int,
                              days_per_week: int = 5) -> None:
        """
        Add a course of radiotherapy with realistic scheduling.
        
        Args:
            start_time: Start time in days
            total_dose: Total radiation dose in Gy
            num_fractions: Number of fractions to divide the dose into
            days_per_week: Treatment days per week (typically 5)
        """
        dose_per_fraction = total_dose / num_fractions
        current_fraction = 0
        current_day = start_time
        
        while current_fraction < num_fractions:
            # Skip weekends
            if current_day % 7 < days_per_week:
                self.schedule.append(TreatmentEvent(
                    time=current_day,
                    treatment_type="radiation",
                    dose=dose_per_fraction,
                    constraints={"cumulative_dose": total_dose}
                ))
                current_fraction += 1
            current_day += 1
            
    def add_chemotherapy_cycle(self,
                             start_time: float,
                             cycle_length: int,
                             num_cycles: int,
                             drug_amount: float,
                             duration: float = 1.0) -> None:
        """
        Add a chemotherapy cycle with rest periods.
        
        Args:
            start_time: Start time in days
            cycle_length: Days per cycle
            num_cycles: Number of cycles to administer
            drug_amount: Amount of drug per administration
            duration: Duration of each administration in days
        """
        for cycle in range(num_cycles):
            time = start_time + cycle * cycle_length
            self.schedule.append(TreatmentEvent(
                time=time,
                treatment_type="chemo",
                dose=drug_amount,
                duration=duration,
                constraints={"min_interval": cycle_length}
            ))
            
    def add_immunotherapy_course(self,
                               start_time: float,
                               interval: float,
                               num_doses: int,
                               boost_factor: float) -> None:
        """
        Add an immunotherapy course.
        
        Args:
            start_time: Start time in days
            interval: Days between doses
            num_doses: Number of doses to administer
            boost_factor: Immune boost factor
        """
        for dose in range(num_doses):
            time = start_time + dose * interval
            self.schedule.append(TreatmentEvent(
                time=time,
                treatment_type="immunotherapy",
                dose=boost_factor,
                constraints={"min_interval": interval}
            ))
            
    def validate_schedule(self) -> bool:
        """
        Validate the treatment schedule against constraints.
        
        Returns:
            bool: True if schedule is valid
        """
        # Sort schedule by time
        self.schedule.sort(key=lambda x: x.time)
        
        # Check constraints
        for i, event in enumerate(self.schedule):
            if event.constraints:
                # Check minimum interval between treatments
                if "min_interval" in event.constraints:
                    min_interval = event.constraints["min_interval"]
                    prev_events = [e for e in self.schedule[:i]
                                 if e.treatment_type == event.treatment_type]
                    if prev_events:
                        time_since_last = event.time - prev_events[-1].time
                        if time_since_last < min_interval:
                            return False
                            
                # Check cumulative dose constraints
                if "cumulative_dose" in event.constraints:
                    treatment_events = [e for e in self.schedule[:i+1]
                                     if e.treatment_type == event.treatment_type]
                    total_dose = sum(e.dose for e in treatment_events)
                    if total_dose > event.constraints["cumulative_dose"]:
                        return False
                        
        return True
        
    def get_treatments_for_timepoint(self, time: float) -> List[TreatmentEvent]:
        """
        Get all treatments that should be applied at a given time point.
        
        Args:
            time: Current simulation time
            
        Returns:
            List of treatment events to apply
        """
        return [event for event in self.schedule
                if abs(event.time - time) < 1e-10]
                
    def optimize_schedule(self,
                        objective_func: callable,
                        constraints: Dict = None) -> None:
        """
        Optimize treatment schedule using simple grid search.
        This is a basic implementation that could be extended with more
        sophisticated optimization methods.
        
        Args:
            objective_func: Function that evaluates schedule quality
            constraints: Additional scheduling constraints
        """
        best_score = float('-inf')
        best_schedule = None
        
        # Simple grid search over schedule parameters
        for shift in np.linspace(-5, 5, 11):  # Try shifting schedule by ±5 days
            # Create temporary schedule
            temp_schedule = [
                TreatmentEvent(
                    time=event.time + shift,
                    treatment_type=event.treatment_type,
                    dose=event.dose,
                    duration=event.duration,
                    constraints=event.constraints
                )
                for event in self.schedule
            ]
            
            # Evaluate schedule
            score = objective_func(temp_schedule)
            
            if score > best_score:
                best_score = score
                best_schedule = temp_schedule
                
        if best_schedule:
            self.schedule = best_schedule