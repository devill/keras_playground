import numpy as np

def choice2d(pmap, count = 10):
    shape = pmap.shape
    indices = np.transpose(np.indices(shape), axes=(1,2,0)).reshape((shape[0]*shape[1],2))
    choice_indices = np.random.choice(len(indices), count, p=pmap.reshape(shape[0]*shape[1]))
    return list(map(lambda x: tuple(x), indices[choice_indices].tolist()))

class LearnedGomokuModel:
    def __init__(self, model):
        self.model = model

    def predict(self, boards):
        pred = self.model.predict(boards, batch_size=64)
        shape = (boards.shape[0], boards.shape[1], boards.shape[2])
        return {
            'outcomes':pred[1].flatten(),
            'action_probability_maps': pred[0].reshape(shape)
        }
