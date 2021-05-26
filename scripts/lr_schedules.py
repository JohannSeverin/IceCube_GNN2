def classic_schedule(lr, buildup = 3, decay = 0.9):

    def lr_schedule():
        # Intial value
        factor = lr * 1 / 2 ** buildup
        yield factor
        
        # Multiply with 2 first few round
        for i in range(buildup):
            factor *= 2
            yield factor

        # Make an exponential decay
        while True:
            factor *= decay
            yield factor

    return lr_schedule


def fast_schedule(lr, buildup = 2, decay = 0.9):

    def lr_schedule():
        # Intial value
        factor = lr * 1 / 2 ** buildup
        yield factor
        
        # Multiply with 2 first few round
        for i in range(buildup):
            factor *= 2
            yield factor

        # Make an exponential decay
        while True:
            factor *= decay
            yield factor

    return lr_schedule