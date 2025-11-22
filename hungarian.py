import numpy as np

def hungarian(A):
    """
    HUNGARIAN Solve the Assignment problem using the Hungarian method.
    
    A - a square cost matrix.
    Returns:
    C - the optimal assignment.
    T - the cost of the optimal assignment.
    """
    orig = A.copy()
    m, n = A.shape
    
    if m != n:
        raise ValueError('HUNGARIAN: Cost matrix must be square!')
    
    # Reduce matrix
    A = hminired(A)
    
    # Do an initial assignment
    A, C, U = hminiass(A)
    
    # Repeat while we have unassigned rows
    while U[n] != 0:
        # Start with no path, no unchecked zeros, and no unexplored rows
        LR = np.zeros(n, dtype=int)
        LC = np.zeros(n, dtype=int)
        CH = np.zeros(n, dtype=int)
        RH = np.zeros(n + 1, dtype=int)
        RH[n] = -1
        
        # No labelled columns
        SLC = []
        
        # Start path in first unassigned row
        r = U[n]
        # Mark row with end-of-path label
        LR[r] = -1
        # Insert row first in labelled row set
        SLR = [r]
        
        # Repeat until we manage to find an assignable zero
        while True:
            # If there are free zeros in row r
            if A[r, n] != 0:
                # get column of first free zero
                l = -A[r, n]
                
                # If there are more free zeros in row r and row r is not yet marked as unexplored
                if A[r, l] != 0 and RH[r] == 0:
                    # Insert row r first in unexplored list
                    RH[r] = RH[n]
                    RH[n] = r
                    
                    # Mark in which column the next unexplored zero in this row is
                    CH[r] = -A[r, l]
            else:
                # If all rows are explored
                if RH[n] <= 0:
                    # Reduce matrix
                    A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                
                # Re-start with first unexplored row
                r = RH[n]
                # Get column of next free zero in row r
                l = CH[r]
                # Advance "column of next free zero"
                CH[r] = -A[r, l]
                # If this zero is last in the list
                if A[r, l] == 0:
                    # remove row r from unexplored list
                    RH[n] = RH[r]
                    RH[r] = 0
            
            # While the column l is labelled, i.e. in path
            while LC[l] != 0:
                # If row r is explored
                if RH[r] == 0:
                    # If all rows are explored
                    if RH[n] <= 0:
                        # Reduce cost matrix
                        A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                    
                    # Re-start with first unexplored row
                    r = RH[n]
                
                # Get column of next free zero in row r
                l = CH[r]
                
                # Advance "column of next free zero"
                CH[r] = -A[r, l]
                
                # If this zero is last in list
                if A[r, l] == 0:
                    # remove row r from unexplored list
                    RH[n] = RH[r]
                    RH[r] = 0
            
            # If the column found is unassigned
            if C[l] == 0:
                # Flip all zeros along the path in LR,LC
                A, C, U = hmflip(A, C, LC, LR, U, l, r)
                # and exit to continue with next unassigned row
                break
            else:
                # else add zero to path
                
                # Label column l with row r
                LC[l] = r
                
                # Add l to the set of labelled columns
                SLC.append(l)
                
                # Continue with the row assigned to column l
                r = C[l]
                
                # Label row r with column l
                LR[r] = l
                
                # Add r to the set of labelled rows
                SLR.append(r)
    
    # Calculate the total cost
    T = np.sum(orig[np.arange(len(C)), C])
    return C, T


def hminired(A):
    """HMINIRED Initial reduction of cost matrix for the Hungarian method"""
    m, n = A.shape
    
    # Subtract column-minimum values from each column
    colMin = np.min(A, axis=0)
    A = A - colMin
    
    # Subtract row-minimum values from each row
    rowMin = np.min(A, axis=1)
    A = A - rowMin[:, np.newaxis]
    
    # Get positions of all zeros
    i, j = np.where(A == 0)
    
    # Extend A to give room for row zero list header column
    A_extended = np.zeros((m, n + 1))
    A_extended[:, :n] = A
    
    for k in range(len(i)):
        row = i[k]
        col = j[k]
        if A_extended[row, n] == 0:
            A_extended[row, n] = -col - 1
        else:
            # Insert pointer
            next_col = -int(A_extended[row, n]) - 1
            A_extended[row, col] = -next_col - 1
            A_extended[row, n] = -col - 1
    
    return A_extended


def hminiass(A):
    """HMINIASS Initial assignment of the Hungarian method"""
    n = A.shape[0]
    
    # Initialize return vectors
    C = np.zeros(n, dtype=int)
    U = np.zeros(n + 1, dtype=int)
    
    # Initialize last/next zero "pointers"
    LZ = np.zeros(n, dtype=int)
    NZ = np.zeros(n, dtype=int)
    
    for i in range(n):
        # Set j to first unassigned zero in row i
        lj = n
        j = -int(A[i, lj]) - 1
        
        # Repeat until we have no more zeros (j==0) or we find a zero in an unassigned column
        while j >= 0 and C[j] != 0:
            # Advance lj and j in zero list
            lj = j
            if A[i, lj] < 0:
                j = -int(A[i, lj]) - 1
            else:
                j = -1
            
            # Stop if we hit end of list
            if j < 0:
                break
        
        if j >= 0:
            # We found a zero in an unassigned column
            # Assign row i to column j
            C[j] = i + 1
            
            # Remove A(i,j) from unassigned zero list
            if lj < n:
                A[i, lj] = A[i, j]
            
            # Update next/last unassigned zero pointers
            if A[i, j] < 0:
                NZ[i] = -int(A[i, j]) - 1
            else:
                NZ[i] = -1
            LZ[i] = lj
            
            # Indicate A(i,j) is an assigned zero
            A[i, j] = 0
        else:
            # We found no zero in an unassigned column
            # Check all zeros in this row
            lj = n
            j = -int(A[i, lj]) - 1 if A[i, lj] < 0 else -1
            
            # Check all zeros in this row for a suitable zero in another row
            found = False
            while j >= 0:
                # Check the in the row assigned to this column
                r = C[j] - 1
                if r >= 0:
                    # Pick up last/next pointers
                    lm = LZ[r]
                    m = NZ[r]
                    
                    # Check all unchecked zeros in free list of this row
                    while m >= 0:
                        # Stop if we find an unassigned column
                        if C[m] == 0:
                            found = True
                            break
                        
                        # Advance one step in list
                        lm = m
                        if A[r, lm] < 0:
                            m = -int(A[r, lm]) - 1
                        else:
                            m = -1
                    
                    if found:
                        # We found a zero in an unassigned column
                        # Replace zero at (r,m) in unassigned list with zero at (r,j)
                        A[r, lm] = -j - 1
                        A[r, j] = A[r, m]
                        
                        # Update last/next pointers in row r
                        if A[r, m] < 0:
                            NZ[r] = -int(A[r, m]) - 1
                        else:
                            NZ[r] = -1
                        LZ[r] = j
                        
                        # Mark A(r,m) as an assigned zero in the matrix
                        A[r, m] = 0
                        
                        # and in the assignment vector
                        C[m] = r + 1
                        
                        # Remove A(i,j) from unassigned list
                        A[i, lj] = A[i, j]
                        
                        # Update last/next pointers in row i
                        if A[i, j] < 0:
                            NZ[i] = -int(A[i, j]) - 1
                        else:
                            NZ[i] = -1
                        LZ[i] = lj
                        
                        # Mark A(i,j) as an assigned zero in the matrix
                        A[i, j] = 0
                        
                        # and in the assignment vector
                        C[j] = i + 1
                        
                        # Stop search
                        break
                
                # Continue with next zero
                lj = j
                if j >= 0 and A[i, lj] < 0:
                    j = -int(A[i, lj]) - 1
                else:
                    j = -1
    
    # Create vector with list of unassigned rows
    rows = C[C != 0] - 1
    r = np.zeros(n, dtype=int)
    r[rows] = rows + 1
    empty = np.where(r == 0)[0]
    
    # Create vector with linked list of unassigned rows
    U = np.zeros(n + 1, dtype=int)
    if len(empty) > 0:
        U[n] = empty[0] + 1
        for idx in range(len(empty) - 1):
            U[empty[idx]] = empty[idx + 1] + 1
    
    return A, C, U


def hmflip(A, C, LC, LR, U, l, r):
    """HMFLIP Flip assignment state of all zeros along a path"""
    n = A.shape[0]
    
    while True:
        # Move assignment in column l to row r
        C[l] = r + 1
        
        # Find zero to be removed from zero list
        # Find zero before this
        m = np.where(A[r, :] == -l - 1)[0]
        if len(m) > 0:
            m = m[0]
            # Link past this zero
            A[r, m] = A[r, l]
            A[r, l] = 0
        
        # If this was the first zero of the path
        if LR[r] < 0:
            # remove row from unassigned row list and return
            U[n] = U[r]
            U[r] = 0
            return A, C, U
        else:
            # Move back in this row along the path and get column of next zero
            l = LR[r]
            
            # Insert zero at (r,l) first in zero list
            A[r, l] = A[r, n]
            A[r, n] = -l - 1
            
            # Continue back along the column to get row of next zero in path
            r = LC[l] - 1


def hmreduce(A, CH, RH, LC, LR, SLC, SLR):
    """HMREDUCE Reduce parts of cost matrix in the Hungarian method"""
    n = A.shape[0]
    
    # Find which rows are covered, i.e. unlabelled
    coveredRows = LR == 0
    
    # Find which columns are covered, i.e. labelled
    coveredCols = LC != 0
    
    r = np.where(~coveredRows)[0]
    c = np.where(~coveredCols)[0]
    
    # Get minimum of uncovered elements
    if len(r) > 0 and len(c) > 0:
        m = np.min(A[np.ix_(r, c)])
        
        # Subtract minimum from all uncovered elements
        A[np.ix_(r, c)] = A[np.ix_(r, c)] - m
        
        # Check all uncovered columns
        for j in c:
            # and uncovered rows in path order
            for i in SLR:
                # If this is a (new) zero
                if A[i, j] == 0:
                    # If the row is not in unexplored list
                    if RH[i] == 0:
                        # insert it first in unexplored list
                        RH[i] = RH[n]
                        RH[n] = i
                        # Mark this zero as "next free" in this row
                        CH[i] = j
                    
                    # Find last unassigned zero on row i
                    row = A[i, :]
                    colsInList = -row[(row < 0) & (row > -n - 1)] - 1
                    if len(colsInList) == 0:
                        # No zeros in the list
                        l = n
                    else:
                        # Find the last one
                        l = colsInList[-1]
                    
                    # Append this zero to end of list
                    A[i, l] = -j - 1
                    A[i, j] = 0
        
        # Add minimum to all doubly covered elements
        r_covered = np.where(coveredRows)[0]
        c_covered = np.where(coveredCols)[0]
        
        if len(r_covered) > 0 and len(c_covered) > 0:
            # Take care of the zeros we will remove
            for i in r_covered:
                for j in c_covered:
                    if A[i, j] <= 0:
                        # Find zero before this in this row
                        lj = np.where(A[i, :] == -j - 1)[0]
                        if len(lj) > 0:
                            lj = lj[0]
                            # Link past it
                            A[i, lj] = A[i, j]
                            # Mark it as assigned
                            A[i, j] = 0
            
            A[np.ix_(r_covered, c_covered)] = A[np.ix_(r_covered, c_covered)] + m
    
    return A, CH, RH

