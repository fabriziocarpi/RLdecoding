import numpy as np

def find_cyclic_auto(llr, k):
    """
    The automorphism group of primitive, narrow-sense binary BCH codes (blocklength N = 2^m-1) includes all permutations of the form
    
        pi_{i,j}(n) = 2^i * n + j    mod N
        
    where i = {0,...,m-1} and n = {0,...,N-1}. For i=0, these are all cyclic shifts. 
        
    """
    N = len(llr)
    m = int(np.log2(N+1))
    if 2**m-1 != N:
        print("length of LLR vector has to be N=2^m-1")
    
    absllr = abs(llr)
    err = np.exp(-absllr)/(1+np.exp(-absllr))
    cap = np.log(2) + err*np.log(err) + (1-err)*np.log(1-err)
    
    # compute fft vector
    kfft = np.fft.fft(np.concatenate([np.ones(np.abs(k)), np.zeros(N-np.abs(k))]))
    if k<0:
        kfft = -kfft
        k = abs(k)
    
    # try all Frobenius automorphisms
    maxval = -1e5
    index = np.arange(N)
    
    for i in range(m):
        test = cap[ (2**i*index)%N  ]
        b = np.real(np.fft.ifft(np.fft.fft(test)*kfft))
        val = b.max()
        ind = (N-(k-1)+b.argmax()+1)%N
        
        if val>maxval:
            p = (2**i*index)%N
            p = p[(index+ind-1)%N]
            maxval = val
    
    return p

def de2bi(d, n, MSBFLAG='right-msb'):
    # de2bi(np.arange(M), m)
    if isinstance(d, (list,)):
        d = np.array(d)
    elif type(d) is np.ndarray:
        d = d.flatten()
    else:
        d = np.array([d])
    power = 2**np.arange(n)
    bitarray = (np.floor((d[:,None]%(2*power))/power))
    if MSBFLAG == 'right-msb':
        return bitarray
    else:
        return np.fliplr(bitarray)

def bi2de(B, MSBFLAG='right-msb'):
    """
    converts a binary vector B to decimal value
    if B is a matrix, conversion is done row-wise
    """
    if(len(B.shape) == 1):
        n = B.shape[0]
        B = np.expand_dims(B, 0)
    elif(len(B.shape) == 2):
        n = B.shape[1]
    else:
        print("this should not happen")

    if MSBFLAG == 'left-msb':
        B = np.fliplr(B)

    power = 2**np.arange(n)
    return (np.sum(B*power, 1)).astype(np.int32)

def gfrref(A):
    """
    Args:
        A: binary matrix

    Returns:
        R: reduced row echolon form of A (over GF(2))
        jb: A[:,jb] is the basis for the range of A, lengh(rb) is the rank of A

    """
    # reduced row echelon form in GF(2)
    (m, n), j = A.shape, 0
    Ar, rank = np.hstack([A, np.eye(m)]), 0
    for i in range(min(m, n)):
        # Find value and index of non-zero element in the remainder of column i.
        while j < n:
            temp = np.where(Ar[i:, j] != 0)[0]
            if len(temp) == 0:
                # If the lower half of j-th row is all-zero, check next column
                j += 1
            else:
                # Swap i-th and k-th rows
                k, rank = temp[0] + i, rank + 1
                if i != k:
                    Ar[[i, k], j:] = Ar[[k, i], j:]
                # Save the right hand side of the pivot row
                pivot = Ar[i, j]
                row = Ar[i, j:].reshape((1, -1)) * pivot**(-1)
                col = np.hstack([Ar[:i, j], [0], Ar[i + 1:, j]]).reshape((-1, 1))
                Ar[:, j:] = (Ar[:, j:] - col * row) % 2
                Ar[i, j:] = Ar[i, j:] * pivot**(-1) % 2
                break
        j += 1
    R = Ar[:, :n]
    col_sum = R.sum(axis=0)
    jb = np.where(col_sum == 1)[0]
    return R, jb
    #perm = np.concatenate([, np.where(col_sum > 1)[0]])
    #R, Y = Ar[:, :n], Ar[:, n:]
    #return R, Y, rank

def gfrank(A):
    _, jb = gfrref(A)
    return len(jb)


def find_rm_auto(llr):
    """
    This function returns an automorphism permutation for RM codes that 
    maps as many as possible of the lowest llrs to positions 0,1,2,4,8...
    """

    if isinstance(llr, (list,)):
        llr = np.array(llr)

    n = llr.shape[0]
    m = np.round(np.log2(n))
    if(n != 2**m):
        raise ValueError("length not power of 2")

    sp = np.argsort(np.abs(llr))

    offset = (de2bi(sp[0], m)).T
    A = ((de2bi(sp[1:], m)).T - offset) % 2 # permutation of all nonzero bit positions
    R, jb = gfrref(A)
    M = A[:, jb]

    # compute permutation
    B = (de2bi(np.arange(n), m)).T
    q = (np.matmul(M,B) + offset) % 2
    p = bi2de(q.T)

    return p
