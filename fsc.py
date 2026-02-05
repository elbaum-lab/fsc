#!/usr/bin/env python3
import numpy as np

def smooth(a):
    b=a.copy()
    for i in range(a.ndim):
        a=a.swapaxes(0,i)
        b=b.swapaxes(0,i)
        b[0]+=a[1]/a.ndim
        b[-1]+=a[-2]/a.ndim
        b[1:-1]+=(a[:-2]+a[2:])/(2.*a.ndim)
        a=a.swapaxes(0,i)
        b=b.swapaxes(0,i)
    return(b)

def checkerboard(a):
    coords=np.ogrid[[slice(s) for s in a.shape]]
    idx1=(sum(coords)%2).astype(bool)
    idx0=np.logical_not(idx1)

    a0=a.copy()
    a1=a.copy()
    a0[idx1]=0
    a1[idx0]=0
    
    return(smooth(a0),smooth(a1))

def fftnfreq(s, d=1.0, sparse=True):
    freqs=[]
    for n, spacing in np.nditer([s,d], casting='no'):
        freqs.append(np.fft.fftfreq(int(n), d=spacing))
    return(np.meshgrid(*freqs, sparse=sparse))

def rfftnfreq(s, d=1.0, sparse=True):
    freqs=None
    for n, spacing in np.nditer([s,d], casting='no'):
        if freqs is None:
            freqs=[]
        else:
            freqs.append(np.fft.fftfreq(n_last, d=d_last))
        n_last,d_last=int(n),spacing
    freqs.append(np.fft.rfftfreq(n_last, d=d_last))
    return(np.meshgrid(*freqs, sparse=sparse))

def compute_fsc(a, vox_size=1.0, bins=100, binsize=None):
    assert bins is None or binsize is None, "Only one of bins and binsize can be specified"
    assert bins is not None or binsize is not None, "One of bins and binsize must be specified"

    a0,a1=checkerboard(a)
    
    a=np.fft.rfftn(a)
    a0=np.fft.rfftn(a0)
    a1=np.fft.rfftn(a1)
    
    numerator=np.real(a0*a1.conj()).flatten()
    a0=np.square(np.abs(a0)).flatten()
    a1=np.square(np.abs(a1)).flatten()

    freqs=rfftnfreq(a.shape, d=vox_size)
    freqs=np.sqrt(sum((np.square(f) for f in freqs))).flatten()
    fsort=np.argsort(freqs)
    
    freqs=freqs[fsort]
    numerator=numerator[fsort]
    a0=a0[fsort]
    a1=a1[fsort]

    if binsize is None:
        binsize = len(freqs) // bins

    pad=binsize-(len(freqs)-1)%binsize-1
    freqs=np.pad(freqs, (0,pad), mode='edge')
    numerator=np.pad(numerator, (0,pad), mode='edge')
    a0=np.pad(a0, (0,pad), mode='edge')
    a1=np.pad(a1, (0,pad), mode='edge')
    
    freqs=freqs.reshape(-1,binsize)
    numerator=numerator.reshape(-1,binsize)
    a0=a0.reshape(-1,binsize)
    a1=a1.reshape(-1,binsize)
    
    
    freqs=np.mean(freqs,axis=1)
    numerator=numerator.sum(axis=1)
    a0=a0.sum(axis=1)
    a1=a1.sum(axis=1)
    denominator=np.sqrt(a0*a1)
    fsc=numerator/denominator

    return(freqs,fsc)



if __name__=="__main__":
    import argparse
    import sys
    import mrcfile

    parser = argparse.ArgumentParser(description="Compute FSC of a 3D volume")
    parser.add_argument("input", nargs='?', type=argparse.FileType('rb'), default=sys.stdin.buffer, help='Input MRC file (stdin)')
    parser.add_argument("--bins", type=int, default=100, help="Number of bins")
    parser.add_argument("--binsize", type=int, default=None, help="Size of bins")
    parser.add_argument("--output", nargs='?', type=argparse.FileType('w'), default="-", help="Output text file")
    args = parser.parse_args()

    mrc=mrcfile.mrcinterpreter.MrcInterpreter(iostream=args.input)
    s=mrc.header
    lengthxyz=[s.cella.x/s.nx, s.cella.y/s.ny, s.cella.z/s.nz]
    axes = [s.mapc, s.mapr, s.maps]          # axes corresp to cols/rows/secs (1,2,3 for X,Y,Z)

    freqs,fsc=compute_fsc(mrc.data.astype(np.float64), vox_size=[lengthxyz[a-1] for a in axes[::-1]], bins=args.bins, binsize=args.binsize)

    for f, v in zip(freqs, fsc):
        args.output.write(f"{f}\t{v}\n")

