import numpy as np


class VAMASBlock:
    
    def __init__(self):
        self.block_id  = None
        self.sample_id = None
        
        self.year         = None
        self.month        = None
        self.day_of_month = None
        self.hours        = None
        self.minutes      = None
        self.seconds      = None
        self.number_of_hours = None
        self.comment         = []
        self.technique       = None
        self.x_coordinate    = None
        self.y_coordinate    = None
        self.value_of_experimental_variable    = []
        self.source_label                      = None
        self.sputtering_ion                    = None
        self.number_of_atoms_in_sputtering_ion = None
        self.sputtering_ion_charge_sign        = None
        self.source_energy                     = None
        self.source_strength                   = None
        self.source_beam_width_x               = None
        self.source_beam_width_y               = None
        self.field_of_view_x                   = None
        self.field_of_view_y                   = None
        
        self.first_linescan_start_x_coordinate  = None
        self.first_linescan_start_y_coordinate  = None
        self.first_linescan_finish_x_coordinate = None
        self.first_linescan_finish_y_coordinate = None
        self.last_linescan_finish_x_coordinate  = None
        self.last_linescan_finish_y_coordinate  = None
        
        self.source_polar_incidence_angle = None
        self.source_azimuth               = None
        self.analyzer_mode                = None
        
        self.pass_energy        = None  # or retard ratio or mass resolution
        self.differential_width = None
        self.lens_magnification = None
        self.analyzer_work_func = None  # eV
        self.acceptance_energy  = None  # eV
        self.target_bias        = None  # volts
        self.analysis_width_x   = None  # micrometers
        self.analysis_width_y   = None  # micrometers
        
        self.analyzer_take_off_polar_angle   = None  # degrees
        self.analyzer_take_off_azimuth_angle = None  # degrees
        self.species_label            = None
        self.transition_label         = None  # or charge state label
        self.detected_particle_charge = None
        
        self.abscissa = {'label': None, 'units': None, 'start': None, 'increment': None}
        self.corresponding_variable = []
        
        self.signal_mode            = None
        self.signal_collection_time = None  # seconds per scan
        self.number_of_scans        = None
        self.signal_time_correction = None  # seconds
        
        self.sputtering_source = {
            'energy':          None,
            'beam_current':    None,
            'beam_width_x':    None,
            'beam_width_y':    None,
            'incidence_angle': None,
            'azimuth_angle':   None,
            'mode':            None,
        }
        
        self.sample_angle = {
            'normal_tilt_polar':   None,
            'normal_tilt_azimuth': None,
            'rotation':            None,
        }
        
        self.future_block = []
        
        self.additional_parameters   = []
        self.min_max_ordinate_values = []
        self.ordinate_values         = []
        
    def __repr__ (self):
        return f"VAMASBlock(block_id='{self.block_id}', sample_id='{self.sample_id}')"


class VAMASFile:
    
    def __init__(self, filename):
        self.blocks = []
        self.read_vamas(filename)
        
    def read_vamas(self, filename):
        with open(filename, 'r') as vms:
            self.format_id           = vms.readline().strip()
            self.institution_id      = vms.readline().strip()
            self.instrument_model_id = vms.readline().strip()
            self.operator_id         = vms.readline().strip()
            self.experiment_id       = vms.readline().strip()
            
            comment_lines_num = int(vms.readline().strip())
            self.comments =  [vms.readline().strip() for _ in range(comment_lines_num)]
            
            self.experiment_mode = vms.readline().strip()
            self.scan_mode       = vms.readline().strip()
            
            if self.experiment_mode in ['MAP', 'MAPDP', 'NORM', 'SDP']:
                self.number_of_spectral_regions = int(vms.readline().strip())
                
            if self.experiment_mode in ['MAP', 'MAPDP']:
                self.number_of_analysis_positions                           = int(vms.readline().strip())
                self.number_of_discrete_x_coordinates_available_in_full_map = int(vms.readline().strip())
                self.number_of_discrete_y_coordinates_available_in_full_map = int(vms.readline().strip())
            
            number_of_experimental_variables = int(vms.readline().strip())
            
            self.experimental_variables = []
            for _ in range(number_of_experimental_variables):
                self.experimental_variables.append(
                    {
                        'label':  vms.readline().strip(),
                        'units':  vms.readline().strip(),
                    }
                )
            
            
            number_of_entries_in_parameter_inclusion_or_exclusion_list = int(vms.readline().strip())
            self.parameter_inclusion_or_exclusion_list = []
            
            for _ in range(number_of_entries_in_parameter_inclusion_or_exclusion_list):
                self.parameter_inclusion_or_exclusion_list.append(vms.readline().strip())
                
            number_of_manually_entered_items_in_block = int(vms.readline().strip())
            self.manual_entered_items_in_block = []
            
            for _ in range(number_of_manually_entered_items_in_block):
                self.manual_entered_items_in_block.append(vms.readline().strip())
                
            number_of_future_upgrade_experiment_entries = int(vms.readline().strip())
            self.number_of_future_upgrade_block_entries = int(vms.readline().strip())
            
            self.future_upgrade_experiment_entry = []
            for _ in range(number_of_future_upgrade_experiment_entries):
                self.future_upgrade_experiment_entry.append(vms.readline().strip())
                
            self.number_of_blocks = int(vms.readline().strip())
            
            for _ in range(self.number_of_blocks):
                vms_block = VAMASBlock()
                
                vms_block.block_id  = vms.readline().strip()
                vms_block.sample_id = vms.readline().strip()
                
                vms_block.year         = int(vms.readline().strip())
                vms_block.month        = int(vms.readline().strip())
                vms_block.day_of_month = int(vms.readline().strip())
                vms_block.hours        = int(vms.readline().strip())
                vms_block.minutes      = int(vms.readline().strip())
                vms_block.seconds      = int(vms.readline().strip())
                vms_block.number_of_hours = int(vms.readline().strip())
                
                lines_num_in_comment = int(vms.readline().strip())
                for _ in range(lines_num_in_comment):
                    line = vms.readline().strip()
                    if '=' in line:
                        line = line.split(' = ')
                    vms_block.comment.append(line)
                
                vms_block.technique = vms.readline().strip()
                
                if self.experiment_mode in ['MAP', 'MAPDP']:
                    vms_block.x_coordinate    = float(vms.readline().strip())
                    vms_block.y_coordinate    = float(vms.readline().strip())
                
                for _ in range(len(self.experimental_variables)):
                    vms_block.value_of_experimental_variable = float(vms.readline().strip())
                
                vms_block.source_label = vms.readline().strip()
                
                if self.experiment_mode in ['MAPDP', 'MAPSVDP', 'SDP', 'SDPSV'] or\
                   vms_block.technique in ['FABMS', 'FABMS energy spec', 'ISS', 
                                           'SIMS','SIMS energy spec', 
                                           'SNMS', 'SNMS energy spec' ]:
                    vms_block.sputtering_ion                    = vms.readline().strip()
                    vms_block.number_of_atoms_in_sputtering_ion = int(vms.readline().strip())
                    vms_block.sputtering_ion_charge_sign        = vms.readline().strip()
                
                vms_block.source_energy       = float(vms.readline().strip())
                vms_block.source_strength     = float(vms.readline().strip())
                vms_block.source_beam_width_x = float(vms.readline().strip())
                vms_block.source_beam_width_y = float(vms.readline().strip())
                
                if self.experiment_mode in ['MAP', 'MAPDP', 'MAPSV', 'MAPSVDP', 'ISEM']:
                    vms_block.field_of_view_x = float(vms.readline().strip())
                    vms_block.field_of_view_y = float(vms.readline().strip())
                    
                if self.experiment_mode in ['MAPSV', 'MAPSVDP', 'SEM']:
                    vms_block.first_linescan_start_x_coordinate  = float(vms.readline().strip())
                    vms_block.first_linescan_start_y_coordinate  = float(vms.readline().strip())
                    vms_block.first_linescan_finish_x_coordinate = float(vms.readline().strip())
                    vms_block.first_linescan_finish_y_coordinate = float(vms.readline().strip())
                    vms_block.last_linescan_finish_x_coordinate  = float(vms.readline().strip())
                    vms_block.last_linescan_finish_y_coordinate  = float(vms.readline().strip())
                
                vms_block.source_polar_incidence_angle = vms.readline().strip()
                vms_block.source_azimuth               = float(vms.readline().strip())
                vms_block.analyzer_mode                = vms.readline().strip()
                vms_block.pass_energy                  = float(vms.readline().strip())
                
                if vms_block.technique == 'AES diff':
                    vms_block.differential_width = float(vms.readline().strip())
                    
                vms.lens_magnification = float(vms.readline().strip())
                
                if vms_block.technique in ['AES', 'ELS', 'ISS', 'UPS', 'XPS']:
                    vms_block.analyzer_work_func = float(vms.readline().strip())
                else:
                    vms_block.acceptance_energy  = float(vms.readline().strip())
                
                vms_block.target_bias      = float(vms.readline().strip())
                vms_block.analysis_width_x = float(vms.readline().strip())
                vms_block.analysis_width_y = float(vms.readline().strip())
                
                vms_block.analyzer_take_off_polar_angle   = float(vms.readline().strip())
                vms_block.analyzer_take_off_azimuth_angle = float(vms.readline().strip()) 
                vms_block.species_label                   = vms.readline().strip()
                vms_block.transition_label                = vms.readline().strip() 
                vms_block.detected_particle_charge        = int(vms.readline().strip())
                
                if self.scan_mode == 'REGULAR':
                    vms_block.abscissa['label']     = vms.readline().strip()
                    vms_block.abscissa['units']     = vms.readline().strip()
                    vms_block.abscissa['start']     = float(vms.readline().strip())
                    vms_block.abscissa['increment'] = float(vms.readline().strip())
                    
                number_of_corresponding_variables = int(vms.readline().strip())
                for _ in range(number_of_corresponding_variables):
                    vms_block.corresponding_variable.append(
                        {
                            'label':  vms.readline().strip(),
                            'units':  vms.readline().strip(),
                            'values': np.array([]),
                        }
                    )
                
                vms_block.signal_mode            = vms.readline().strip()
                vms_block.signal_collection_time = float(vms.readline().strip())
                vms_block.number_of_scans        = int(vms.readline().strip())
                vms_block.signal_time_correction = float(vms.readline().strip())
                
                if vms_block.technique in  [ 'AES diff', 'AES dir', 'EDX', 'ELS', 'UPS', 'XPS', 'XRF'] and\
                   self.experiment_mode in ['MAPDP', 'MAPSVDP', 'SDP', 'SDPSV']:
                    vms_block.sputtering_source['energy']          = float(vms.readline().strip())
                    vms_block.sputtering_source['beam_width_x']    = float(vms.readline().strip())
                    vms_block.sputtering_source['beam_width_y']    = float(vms.readline().strip())
                    vms_block.sputtering_source['incidence_angle'] = float(vms.readline().strip())
                    vms_block.sputtering_source['azimuth_angle']   = float(vms.readline().strip())
                    vms_block.sputtering_source['mode']            = float(vms.readline().strip())
                    
                vms_block.sample_angle['normal_tilt_polar']   = float(vms.readline().strip())
                vms_block.sample_angle['normal_tilt_azimuth'] = float(vms.readline().strip())
                vms_block.sample_angle['rotation']            = float(vms.readline().strip())
                
                number_of_additional_numerical_parameters = int(vms.readline().strip())
                for _ in range(number_of_additional_numerical_parameters):
                    vms_block.additional_parameters.append(
                        {
                            'label': vms.readline().strip(),
                            'units': vms.readline().strip(),
                            'value': float(vms.readline().strip()),
                        }
                    )
                    
                for _ in range(self.number_of_future_upgrade_block_entries):
                    vms_block.future_block.append(vms.readline().strip())
                
                number_of_ordinate_values = int(vms.readline().strip())
                
                for _ in range(number_of_corresponding_variables):
                    vms_block.min_max_ordinate_values.append(
                        {
                            'minimum': float(vms.readline().strip()),
                            'maximum': float(vms.readline().strip()),
                        }
                    )
                
                for _ in range(number_of_ordinate_values // number_of_corresponding_variables):
                    for i in range(number_of_corresponding_variables):
                        vms_block.corresponding_variable[i]['values'] = np.append(
                            vms_block.corresponding_variable[i]['values'], float(vms.readline().strip())
                        )
                
                self.blocks.append(vms_block)