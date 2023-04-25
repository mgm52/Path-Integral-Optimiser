class BaseTask:
    def dataset(self):
        raise NotImplementedError

    def loss(self, y, GT):
        # y is of shape (trajectories_in_batch, samples_in_batch, GTSIZE)
        # GT is of shape (samples_in_batch, GTSIZE)
        # return shape (trajectories_in_batch)
        raise NotImplementedError

    def datasize(self):
        raise NotImplementedError
    
    def gtsize(self):
        raise NotImplementedError