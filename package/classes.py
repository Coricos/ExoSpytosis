# Author:  Meryll Dindin
# Date:    10/26/2019
# Project: ExoSpytosis

try: from package.utils import *
except: from utils import *

class VideoChunk:
    
    DIM = (256, 256)
    
    def __init__(self, filename='data-qbi-hackathon/video.nd2', graph=False):
        
        self.pth = filename
        # Extract attributes
        with ND2Reader(self.pth) as fle:
            self.cnt = fle.metadata['num_frames']
            self.rte = fle.metadata['experiment']['loops'][0]['sampling_interval']
        # Run the initialization
        self.initialize(graph=graph)      
 
    def initialize(self, chunksize=3500, lam=1e9, p=0.01, graph=False):
        
        self.mus = np.zeros(self.cnt)
        # Aggregate the general evolution of the means
        with ND2Reader(self.pth) as fle: 
            for idx in range(self.cnt): 
                self.mus[idx] = np.mean(np.asarray(fle[idx], dtype=np.uint8))
        # Build the relative baseline
        self.bsl = baseline(self.mus, 1e9, 0.001)
        
        if graph:
            
            plt.figure(figsize=(18,4))
            plt.plot(self.mus, color='crimson', label='Raw Serie - Means per Frame')
            plt.plot(self.bsl, color='black', label='Estimated Baseline from Raw Serie')
            plt.legend(loc='upper right')
            plt.yticks([])
            plt.show()
    
        self.msk = np.zeros((256, 256, chunksize), dtype=np.uint8)
        # Instantiate the mask
        with ND2Reader(self.pth) as fle:
            for idx in range(chunksize):
                img = np.asarray(fle[idx], dtype=np.uint8)
                img[img <= self.bsl[idx]] = self.bsl[idx]
                self.msk[:,:,idx] = (img - self.bsl[idx]).astype('int')
        
        if graph:
            
            plt.figure(figsize=(18,7))
            plt.subplot(1,2,1)
            plt.title('STD Heatmap Initialization')
            sns.heatmap(np.std(self.msk, axis=-1))
            plt.subplot(1,2,2)
            plt.title('MAX Heatmap Initialization')
            sns.heatmap(np.max(self.msk, axis=-1))
            plt.tight_layout()
            plt.show()

        # Extract the mask looked for
        self.msk = np.max(self.msk, axis=-1)
    
    def get_frame(self, index):
    
        with ND2Reader(self.pth) as fle: 

            return np.asarray(fle[index], dtype=np.uint8) - self.bsl[index]
    
    def get(self, start_slice, chunksize=512, alpha=2.0):
        
        chunk = np.zeros((*self.DIM, chunksize))
        inits = np.zeros((*self.DIM, chunksize))
        
        with ND2Reader(self.pth) as fle: 
            for idx in range(chunksize):
                img = np.asarray(fle[idx+start_slice], dtype=np.uint8)
                inits[:,:,idx] = img
                img[img <= self.bsl[idx+start_slice]] = self.bsl[idx+start_slice]
                img = img - self.bsl[idx+start_slice] - self.msk
                img[img < 0] = 0
                chunk[:,:,idx] = img.astype('int')
    
        return inits, chunk

class Point:
    
    def __init__(self, x, y, z):
        
        self.x = x
        self.y = y
        self.z = z
        
    def distance(self, point):
        
        return np.sqrt((point.x - self.x)**2 + (point.y - self.y)**2 + (point.z - self.z)**2)

class Event:
    
    def __init__(self, origin, layer):
        
        self.pts = [origin]
        self.lyr = layer
    
    def distance(self, point):
        
        return np.min([pts.distance(point) for pts in self.pts])
    
    def fusion(self, event):
        
        self.pts += event.pts
        self.lyr = max(self.lyr, event.lyr)
        
    def duration(self):
        
        dur = [point.z for point in self.pts]
        return np.max(dur) + 1 - np.min(dur)
       
class Extractor:
    
    def __init__(self, threshold, min_distance):
        
        self.thr = threshold
        self.m_d = min_distance
        
    @staticmethod
    def acquisition_distance(event_1, event_2):
    
        return np.min([event_2.distance(point) for point in event_1.pts])
    
    def from_3d_chunk(self, chunk, layer):
        
        # Extract non-lonely points
        x, y, z = np.where(chunk > self.thr[layer])
        z += layer*chunk.shape[2]
        pts = np.vstack((x, y, z)).T
        
        if len(pts) < 2: 
            return []
        
        else:
            # Build the graph of connections
            kdt = KDTree(pts, leaf_size=3, metric='euclidean')
            dst = kdt.query(pts[:,:], k=2)[0]
            dst = np.asarray([d[1] for d in dst])
            msk = dst < self.m_d[layer]

            # Sort the points based on their temporal dimension
            x, y, z = x[msk], y[msk], z[msk]
            idx = np.argsort(z)
            x, y, z = x[idx], y[idx], z[idx]

            # Build the initial list of events
            events = []
            for u, v, w in zip(x, y, z):

                if len(events) == 0: 
                    events.append(Event(Point(u, v, w), layer))

                else:
                    pts = Point(u, v, w)
                    evt = sorted(events, key=lambda x: x.distance(pts))
                    if evt[0].distance(pts) > self.m_d[layer]: events.append(Event(pts, layer))
                    else: evt[0].pts.append(pts)

            return [e for e in events if (e.duration() > 1) & (len(e.pts) > 5)]
        
    def fusion_events(self, events, alpha=2.0):
        
        if len(events) == 0:
            return []
                    
        else:           
            # Apply an efficient layer-based mask
            lyr = np.asarray([e.lyr for e in events])
            msk = np.where(lyr >= max(lyr)-1)[0]
            
            # Compute the pairwise distance matrix
            mat = np.zeros((len(events), len(events)))
            for i in range(len(events)):
                for j in range(i+1, len(events)):
                    if (i in msk) and (j in msk):
                        mat[i,j] = np.min([events[i].distance(point) for point in events[j].pts])

            # Refilter the points based on their relationships
            idx = np.vstack(np.where((mat < alpha*self.m_d[max(lyr)]) & (mat > 0.0)))
            idx = idx[:,idx[0,:] < idx[1,:]]

            # Fusion events if they turn out to be close enough
            fusions = []
            for i in range(idx.shape[1]): 
                if len(fusions) == 0: 
                    fusions.append(list(idx[:,i]))
                else:
                    boo = False
                    i,j = idx[:,i]
                    for fusion in fusions: 
                        if (i in fusion) or (j in fusion):
                            fusion += [i, j]
                            boo = True
                            break
                    if not boo: fusions.append([i, j])
            
            # Apply final filtering
            fusions = [list(np.unique(fusion)) for fusion in fusions]
            for fusion in fusions: _ = [events[fusion[0]].fusion(events[i]) for i in fusion[1:]]
            evt, fus = [], []
            for fusion in fusions: fus += fusion[1:]
            for i, e in enumerate(events):
                if i not in fus: evt.append(e)

            return evt
    
    def display_3d_events(self, events):
        
        if len(events) == 0: 
            return None
        
        else:
            pts, cls, roi = [], [], 1
            for event in events:
                for point in event.pts: 
                    pts.append([point.x, point.y, point.z])
                    cls.append(roi)
                roi += 1

            pts, cls = np.asarray(pts), np.asarray(cls)
            x, y, z = pts[:,0], pts[:,1], pts[:,2]

            arg = {'mode': 'markers', 'marker': dict(size=4, color=cls)}
            fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, **arg)])
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            fig.show()
