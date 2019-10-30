# Author:  Meryll Dindin
# Date:    10/26/2019
# Project: ExoSpytosis

try: from package.utils import *
except: from utils import *

# Multiprocessing related wrappers

def mp_noiselevel(index, filename, percentile):
    
    with ND2Reader(filename) as fle: img = np.asarray(fle[index])
    return np.percentile(Frame(img).getBackground(), percentile)

def mp_get_frames(index, filename):
    
    with ND2Reader(filename) as fle: return Frame(np.asarray(fle[index])).img
    
def mp_extraction(index, filename, noise, background):
    
    with ND2Reader(filename) as fle: frm = Frame(np.asarray(fle[index]))
    x,y = np.where(frm.applyFilters(noise, background))
    return np.vstack((x, y, np.full(len(x), index))).T

# General classes

class Frame:
    
    def __init__(self, image):
        
        self.img = np.clip(image, 0, 255) / 255.0
        self.dim = image.shape
        
    def getMasks(self, threshold=0.1):
        
        sig = estimate_sigma(self.img, average_sigmas=False, multichannel=False)
        blr = snd.gaussian_filter(self.img, sigma=sig)
        thr = threshold_otsu(blr)
        bry = (blr >= thr)
        new = binary_opening(bry)
        
        new = snd.morphology.binary_fill_holes(new)
        mat = np.array([[0, 0, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0]])
        msk = binary_dilation(new, selem=mat)
        
        warnings.simplefilter('ignore')
        msk = rank.mean(msk.astype('float'), selem=disk(16))
        out = np.where(msk > threshold, 0, 1).astype('bool')
        ins = np.where(msk <= threshold, 0, 1).astype('bool')

        # Memory efficiency
        del sig, blr, thr, bry, msk
              
        return ins, out
     
    def getBackground(self, threshold=0.1):
        
        # Retrieve the decomposition
        _,m = self.getMasks()
        # Apply the mask
        return self.img[m]
    
    def maskOnNoise(self, noise_level):
        
        return np.where(self.img < noise_level, 0, 1)
    
    def maskOnBackground(self, background):
        
        return np.where(self.img < background, 0, 1)
        
    def applyFilters(self, noise, background, threshold=0.1, sizes=(2,3,1,3)):

        # Look for the masks
        m,_ = self.getMasks(threshold=threshold)
        m_0 = self.maskOnNoise(noise)
        m_1 = self.maskOnBackground(background)
    
        # Override image
        self.img = (self.img*m*m_0*m_1 * 255.0).astype(np.uint8)
        
        # Post-process the results
        self.img = binary_dilation(self.img, selem=disk(sizes[0]))
        self.img = binary_erosion(self.img, selem=disk(sizes[1]))
        self.img = gaussian(self.img, sigma=sizes[2])
        self.img = binary_erosion(self.img, selem=disk(sizes[3]))
        
        return np.where(self.img > 1e-2, 1, 0).astype(np.uint8)

class Video:
    
    def __init__(self, filename, verbose=False, max_threads=cpu_count()):
        
        self.vrb = verbose
        self.pth = filename
        self.cpu = max_threads
        # Extract attributes
        with ND2Reader(self.pth) as fle:
            self.cnt = fle.metadata['num_frames']
            self.rte = fle.metadata['experiment']['loops'][0]['sampling_interval']
        
        # Run the initialization
        self._estimate_noiselevel()
        self._estimate_background()

    def _estimate_noiselevel(self, pad=50, percentile=99):
        
        t_0 = time.time()
        fun = partial(mp_noiselevel, filename=self.pth, percentile=percentile)

        if self.cpu == 1:
            res = np.asarray([fun(idx) for idx in np.arange(0, self.cnt, pad)])
        else:
            with Pool(processes=self.cpu) as pol:
                res = np.asarray(pol.map(fun, np.arange(0, self.cnt, pad)))
                pol.close()
            
        self.lvl = np.mean(res)

        if self.vrb: print('# Noise level extracted in {} seconds'.format(np.round(time.time()-t_0, 3)))
        
        # Memory efficiency
        del fun, res, pol
        
    def _estimate_background(self, frames=(1000, 3000, 10), percentile=90):

        t_0 = time.time()
        fun = partial(mp_get_frames, filename=self.pth)

        if self.cpu == 1:
            res = [fun(idx) for idx in np.arange(*frames)]
        else:
            with Pool(processes=self.cpu) as pol:
                res = pol.map(fun, np.arange(*frames))
                pol.close()
            
        warnings.simplefilter('ignore')
        self.bkg = np.percentile(res, percentile, axis=0)
        m_x = self.bkg.max()
        self.bkg = rank.mean(self.bkg, selem=disk(10)) / 255.0
        self.bkg = self.bkg * (m_x / self.bkg.max())

        if self.vrb: print('# Static background estimated in {} seconds'.format(np.round(time.time()-t_0, 3)))
        
        # Memory efficiency
        del fun, res, pol
    
    def visualizeFiltering(self, index, threshold=0.1):
        
        with ND2Reader(self.pth) as fle: frm = Frame(np.asarray(fle[index]))
        
        # Retrieve the decomposition
        _,m = frm.getMasks(threshold=threshold)
        m_0 = frm.maskOnNoise(self.lvl)
        m_1 = frm.maskOnBackground(self.bkg)
        
        arg = {'vmin': 0, 'vmax': 1}
        
        plt.figure(figsize=(18,4))
        plt.subplot(1,5,1)
        plt.title('Initial Image')
        plt.imshow(frm.img, **arg)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,5,2)
        plt.title('Estimate Contours')
        plt.imshow(m, **arg)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,5,3)
        plt.title('Noise Thresholding')
        plt.imshow(m_0, **arg)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,5,4)
        plt.title('Background Thresholding')
        plt.imshow(m_1, **arg)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1,5,5)
        plt.title('Post-processed Masks')
        plt.imshow(frm.applyFilters(self.lvl, self.bkg), **arg)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        
    def process(self, max_distance=3):
        
        t_0 = time.time()
        fun = partial(mp_extraction, filename=self.pth, noise=self.lvl, background=self.bkg)
        
        if self.cpu == 1:
            res = [fun(idx) for idx in np.arange(self.cnt)]
        else:
            with Pool(processes=self.cpu) as pol:
                res = pol.map(fun, np.arange(self.cnt))
                pol.close()
        pts = np.vstack(res)

        if self.vrb: print('# Points of interest extracted in {} seconds'.format(np.round(time.time()-t_0, 3)))
        t_0 = time.time()
        
        # Run density-based clustering
        cls = DBSCAN(eps=max_distance, n_jobs=self.cpu).fit_predict(pts)

        if self.vrb: print('# Event clustering in {} seconds'.format(np.round(time.time()-t_0, 3)))
        
        # First event estimation
        list_events = []    
        for crd in np.unique(cls):
            list_points = [Point(*c) for c in pts[np.where(cls == crd)[0],:]]
            list_events.append(Event(list_points, crd))
            
        # Memory efficiency
        del fun, res, pol, pts, cls
            
        return list_events

class Point:
    
    def __init__(self, x, y, z):
        
        self.x = x
        self.y = y
        self.z = z
        
    def distance(self, point):
        
        return np.sqrt((point.x - self.x)**2 + (point.y - self.y)**2) + np.abs(point.z - self.z)
    
class Event:
    
    def __init__(self, points, roi):
        
        self.roi = roi
        self.pts = points
    
    def distance(self, event):
        
        dis = []
        for point in self.pts:
            dis.append(np.min([pts.distance(point) for pts in self.pts]))
        return np.min(dis)
    
    def fusion(self, event):
        
        self.pts += event.pts
        
    def getPoints(self):
        
        try: return np.vstack([[p.x, p.y, p.z, self.roi] for p in self.pts])
        except: return None
        
    def duration(self):
        
        dur = [point.z for point in self.pts]
        return np.max(dur) + 1 - np.min(dur)  
    
    def getAreaVolume(self):
        
        slc = np.vstack([[p.x, p.y, p.z] for p in self.pts])
        
        if len(np.unique(slc[:,2])) == 1: 
            ull = ConvexHull(points=slc[:,:2])
        else: 
            ull = ConvexHull(points=slc[:,:])
            
        return (ull.area, ull.volume)
    
    def focus(self):
        
        disk_points = []
        mat = self.getPoints()
        
        x = int(np.round(np.median(mat[:,0])))
        y = int(np.round(np.median(mat[:,1])))
        self.cen = (x, y)

        for u_z in np.unique(mat[:,2]):
            u,v = np.meshgrid(np.arange(x-2, x+3), np.arange(y-2, y+3))
            m_z = np.full(len(u), u_z).reshape(-1,1)
            for u_x in u:
                for u_y in v: 
                    disk_points += list(np.hstack((u_x.reshape(-1,1), u_y.reshape(-1,1), m_z)))

        self.pts = [Point(*p) for p in disk_points]

class EventManager:
    
    def __init__(self, events):
        
        self.evt = events
        
    def filterAreaVolume(self, min_area=10, min_volume=10):
        
        list_evt = []
        for event in self.evt: 
            a,v = event.getAreaVolume()
            if (a > min_area) and (v > min_volume): list_evt.append(event)
        self.evt = list_evt
        
    def focusAreas(self):
        
        for event in self.evt: event.focus()
        
    def display(self):
        
        points = np.vstack([e.getPoints() for e in self.evt if len(e.pts) > 0])
        
        x,y,z = points[:,0], points[:,1], points[:,2]
        arg = {'mode': 'markers', 'marker': dict(size=3, color=points[:,3])}
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, **arg)])
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
