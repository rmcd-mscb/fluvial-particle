class settings(dict):
    """ Handler class to user defined parameters. Allows us to check a users input parameters in the backend """

    def __init__(self, **kwargs):
        missing = [ x for x in self.required_keys if not x in kwargs ]
        if len(missing) > 0:
            raise ValueError("Missing {} from the user parameter file".format(missing))

        for key, value in kwargs.items():
            self[key] = value

    @property
    def required_keys(self):
        return ('SimTime',
                'dt',
                'min_depth',
                'beta_x',
                'beta_y',
                'beta_z', 
                'Track2D',
                'Track3D',
                'PrintAtTick',
                'file_name_2da',
                'file_name_3da',
                'NumPart',
                'StartLoc',
                'amplitude',
                'period',
                'min_elev'
                )

    @classmethod
    def read(cls, filename):
        options = {}
        # Load user parameters
        with open(filename, 'r') as f:
            f = '\n'.join(f.readlines())
            exec(f, {}, options)

        return cls(**options)



# def printtimes():
#     """[summary].

#     Returns:
#         [type]: [description]
#     """
#     step = CheckAtTick * delta
#     return range(int(step), int(SimTime) + 1, int(step))

