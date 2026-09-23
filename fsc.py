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

    orig_shape = a1.shape
    if np.ndim(vox_size) == 0:
        vox_size = [vox_size]

    freqs=rfftnfreq(orig_shape, d=vox_size)
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


    if (binsize is None) == (bins is None):
        raise AssertionError("Specify exactly one of bins or binsize")

    target_pop = len(freqs) // bins if binsize is None else binsize
    
    fundamental_step = 1.0 / (max(orig_shape) * max(vox_size)) 
    min_width = 3.0 * fundamental_step  
    
    # high-resolution 1D histogram
    fine_bins = max(int(np.max(freqs) / (fundamental_step / 4.0)), 10)
    fine_counts, fine_edges = np.histogram(freqs, bins=fine_bins)

    # dynamically determine actual bin edges
    bin_edges = [0.0]
    current_pop = 0

    for edge, count in zip(fine_edges[1:], fine_counts):
        current_pop += count
        if current_pop >= target_pop and (edge - bin_edges[-1]) >= min_width:
            bin_edges.append(edge)
            current_pop = 0

    bin_edges[-1] = fine_edges[-1] + 1e-5 

    # assign voxels to dynamic bins and accumulate
    bin_indices = np.digitize(freqs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    num_bins = len(bin_edges) - 1

    num_sum = np.bincount(bin_indices, weights=numerator, minlength=num_bins)
    a1_sum = np.bincount(bin_indices, weights=a1, minlength=num_bins)
    a2_sum = np.bincount(bin_indices, weights=a2, minlength=num_bins)
    voxel_counts = np.bincount(bin_indices, minlength=num_bins)

    # calculate final binned frequencies and FSC
    valid_bins = voxel_counts > 0
    freqs_binned = np.zeros(num_bins, dtype=np.float64)
    freqs_binned[valid_bins] = np.bincount(bin_indices, weights=freqs, minlength=num_bins)[valid_bins] / voxel_counts[valid_bins]

    denominator = np.sqrt(a1_sum * a2_sum)
    result = np.zeros(num_bins, dtype=np.float64)
    valid_denom = denominator > 0
    result[valid_denom] = num_sum[valid_denom] / denominator[valid_denom]

    return freqs_binned[valid_bins], result[valid_bins]


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

