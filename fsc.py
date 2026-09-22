#!/usr/bin/env python3
import numpy as np

def smooth(a):
    """Smooth an array by averaging nearest neighbors along each axis."""
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
    """Split an array into even- and odd-parity checkerboard smoothed half-maps."""
    coords=np.ogrid[[slice(s) for s in a.shape]]

    idx=np.asarray(sum(coords)%2, dtype=bool)
    a0=a.copy()
    a0[idx]=0
    a0=smooth(a0)

    idx=np.logical_not(idx)
    a1=a.copy()
    a1[idx]=0
    a1=smooth(a1)

    return(a0, a1)

def fftnfreq(s, d=1.0, sparse=True):
    """Return FFT frequency grids for all axes of a real-valued array."""
    freqs=[]
    for n, spacing in np.nditer([s,d], casting='no'):
        freqs.append(np.fft.fftfreq(int(n), d=spacing))
    return(np.meshgrid(*freqs, sparse=sparse, indexing='ij'))

def rfftnfreq(s, d=1.0, sparse=True):
    """Return FFT frequency grids with a real FFT axis on the last dimension."""
    freqs=[]
    it = np.nditer([s, d], casting='no')
    try:
        n_prev, d_prev = next(it)
    except StopIteration as exc:
        raise ValueError("shape must have at least one axis") from exc

    for n, spacing in it:
        freqs.append(np.fft.fftfreq(int(n_prev), d=float(d_prev)))
        n_prev, d_prev = n, spacing

    freqs.append(np.fft.rfftfreq(int(n_prev), d=float(d_prev)))
    return(np.meshgrid(*freqs, sparse=sparse, indexing='ij'))

def compute_fsc(a, **kwargs):
    """Compute FSC for a volume by splitting it into checkerboard half-maps."""
    a0,a1=checkerboard(a)
    return(fsc2(a0, a1, nyquist=0.25, **kwargs))

def fsc2(a1, a2, vox_size=1.0, bins=100, binsize=None, nyquist=0.5, full=False):
    """Compute the Fourier shell correlation between two equally shaped volumes."""
    """non-uniform binning to keep equal variance"""
    assert a1.shape==a2.shape, "Input volumes must have the same shape"

    if np.ndim(vox_size) == 0:
        vox_size = [vox_size]

    freqs=rfftnfreq(a1.shape, d=vox_size)
    freqs=np.sqrt(sum((np.square(f) for f in freqs))).flatten()

    a1=np.fft.rfftn(a1).flatten()
    a2=np.fft.rfftn(a2).flatten()

    if not full:
        mask=freqs < (nyquist/max(vox_size))
        freqs=freqs[mask]
        a1=a1[mask]
        a2=a2[mask]

    numerator=np.real(a1*a2.conj())
    a1=np.square(np.abs(a1))
    a2=np.square(np.abs(a2))

    fsort=np.argsort(freqs)
    freqs=freqs[fsort]
    numerator=numerator[fsort]
    a1=a1[fsort]
    a2=a2[fsort]

    if (binsize is None) == (bins is None):
        raise AssertionError("Specify exactly one of bins or binsize")
    if binsize is None:
        binsize = len(freqs) // bins

    remainder = len(freqs) % binsize
    if remainder != 0:
        freqs = freqs[:-remainder]
        numerator = numerator[:-remainder]
        a1 = a1[:-remainder]
        a2 = a2[:-remainder]
    
    freqs=freqs.reshape(-1,binsize)
    numerator=numerator.reshape(-1,binsize)
    a1=a1.reshape(-1,binsize)
    a2=a2.reshape(-1,binsize)    
    
    freqs=np.mean(freqs,axis=1)
    numerator=numerator.sum(axis=1)
    a1=a1.sum(axis=1)
    a2=a2.sum(axis=1)
    denominator=np.sqrt(a1*a2)
    result=numerator/denominator

    return(freqs, result)



if __name__=="__main__":
    import argparse
    import sys
    import mrcfile.mrcinterpreter

    parser = argparse.ArgumentParser(description="Compute FSC of a 3D volume or between two volumes")
    parser.add_argument("input", nargs='*', type=argparse.FileType('rb'), help='Input MRC file(s) (stdin if omitted)')
    parser.add_argument("--bins", type=int, default=100, help="Number of bins")
    parser.add_argument("--binsize", type=int, default=None, help="Size of bins")
    parser.add_argument("--output", nargs='?', type=argparse.FileType('w'), default="-", help="Output text file")
    parser.add_argument("--full", action='store_true', help="Compute full FSC (BEWARE)")
    args = parser.parse_args()

    # Default to stdin if no arguments are passed
    if not args.input:
        args.input = [sys.stdin.buffer]

    if len(args.input) > 2:
        parser.error("A maximum of two input files are supported.")

    mrc1 = mrcfile.mrcinterpreter.MrcInterpreter(iostream=args.input[0])
    s1 = mrc1.header
    lengthxyz = [s1.cella.x/s1.nx, s1.cella.y/s1.ny, s1.cella.z/s1.nz]
    axes = [s1.mapc, s1.mapr, s1.maps]          # axes corresp to cols/rows/secs (1,2,3 for X,Y,Z)
    vox_size = [lengthxyz[a-1] for a in axes[::-1]]

    data1 = mrc1.data.astype(np.float64)

    if len(args.input) == 1:
        freqs, corrs = compute_fsc(data1, vox_size=vox_size, bins=args.bins, binsize=args.binsize, full=args.full)
    else:
        mrc2 = mrcfile.mrcinterpreter.MrcInterpreter(iostream=args.input[1])
        data2 = mrc2.data.astype(np.float64)
        freqs, corrs = fsc2(data1, data2, vox_size=vox_size, bins=args.bins, binsize=args.binsize, full=args.full)

    for f, v in zip(freqs, corrs):
        args.output.write(f"{f}\t{v}\n")

