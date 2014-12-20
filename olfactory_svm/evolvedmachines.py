import numpy as np

def enum(**enums):
    return type('Enum', (), enums)

class RSA:
    def __init__(self, latencyScale, sigmoidRate, normalizeSpikes, maxLatency, maxSpikes):
        self.latencyScale = latencyScale
        self.sigmoidRate = sigmoidRate
        self.maxSpikes = maxSpikes
        self.maxLatency = maxLatency
        self.normalizeSpikes = normalizeSpikes
        self.desensitize = False
    
    def spikeLatencies(self, X):
        if (self.desensitize):
            latency = X
        else:
            if (self.sigmoidRate):
                rate = 1.0/(1.0 + np.exp(-(X-0.5)*10.0))
                latency = self.latencyScale / rate
            else:
                latency = self.latencyScale / X
        # find all values greater than maxL and set to maxL
        maxed = latency > self.maxLatency
        latency[maxed] = self.maxLatency
        return latency
        
    def countNspikes(self, X):
        #round latency to nearest int and set 0 <- 1        
        latency = np.round(self.spikeLatencies(X), 0)
        minL = latency <= 0.0
        latency[minL] = 1.0
        
        #intergrationWindow = np.ones(latency.shape[0])
        totalSpikes = np.zeros(latency.shape[0])
        spikeTrain = np.zeros(latency.shape)
        
        #original implementation was triple-nested, needs to be vectorized
        #this was simple enough for the inner loop, use bitmasking to
        #check if a spike is emitted on each tstep, and that its latency != max
        #for the outer loop this is a little tougher, how can we achieve
        #a spike limit w/o actually stepping through the funciton? it may
        #be easier to just do all the preprocessing in the fluid sim...
        for i in range(latency.shape[0]):
            for j in range(self.maxLatency):
                spikeEmitted = ((j + 1) % latency[i] == 0)
                notMax = latency[i] < self.maxLatency
                
                spikeIdx = np.bitwise_and(spikeEmitted, notMax)
                
                spikeTrain[i][spikeIdx] += 1
                totalSpikes[i] += np.sum(spikeIdx)
                if (totalSpikes[i] >= self.maxSpikes): break

        if (self.normalizeSpikes):
            spikeScale = np.ones(totalSpikes.shape)
            spikesEmitted = totalSpikes > 0.0
            spikeScale[spikesEmitted] = self.maxSpikes / totalSpikes[spikesEmitted]
            spikeTrain = np.transpose(spikeScale * np.transpose(spikeTrain))
            
        return spikeTrain