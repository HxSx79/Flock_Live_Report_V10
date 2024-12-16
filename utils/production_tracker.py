from datetime import datetime
from typing import Dict, Optional
from .bom_reader import BOMReader
from .excel_logger import ExcelLogger

class ProductionTracker:
    def __init__(self):
        self.bom_reader = BOMReader()
        self.excel_logger = ExcelLogger()
        self.line_data = {
            'line1': {
                'part': {'program': '', 'number': '', 'description': ''},
                'production': {'quantity': 0}
            },
            'line2': {
                'part': {'program': '', 'number': '', 'description': ''},
                'production': {'quantity': 0}
            }
        }

    def update_production(self, counts: Dict[str, int], latest_crossings: Dict[str, Optional[Dict]]) -> None:
        """Update production data based on line crossings"""
        for line_key in ['line1', 'line2']:
            # Update counts
            self.line_data[line_key]['production']['quantity'] = counts[line_key]
            
            # Process new crossings
            if latest_crossings[line_key]:
                class_name = latest_crossings[line_key]['class_name']
                
                # Get part info from BOM
                part_info = self.bom_reader.get_part_info(class_name)
                
                # Update line data
                self.line_data[line_key]['part'].update({
                    'program': part_info['program'],
                    'number': part_info['part_number'],
                    'description': part_info['description']
                })
                
                # Log to Excel
                self.excel_logger.log_crossing(
                    line_number=1 if line_key == 'line1' else 2,
                    class_name=class_name,
                    part_info=part_info
                )

    def get_all_data(self) -> Dict:
        """Get all production data for display"""
        return {
            'line1_part': self.line_data['line1']['part'],
            'line1_production': self.line_data['line1']['production'],
            'line2_part': self.line_data['line2']['part'],
            'line2_production': self.line_data['line2']['production'],
            'total_quantity': sum(data['production']['quantity'] 
                                for data in self.line_data.values())
        }