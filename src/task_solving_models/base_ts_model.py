class BaseTSModel:
    def __init__(self, DATASIZE, GTSIZE):
        self.DATASIZE = DATASIZE
        self.GTSIZE = GTSIZE

    def forward(self, x, w):
        # x is of shape (samples_in_batch, DATASIZE)
        # w is of shape (trajectories_in_batch, WEIGHTSIZE)
        # return shape (trajectories_in_batch, samples_in_batch, GTSIZE)
        raise NotImplementedError
    
    def param_size(self):
        raise NotImplementedError